from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional, Tuple, Union, Dict
import scipy.sparse
import numpy as np
import pandas as pd
from toponymy.keyphrases import (
    central_keyphrases,
    information_weighted_keyphrases,
    bm25_keyphrases,
    submodular_selection_information_keyphrases,
)
from toponymy.exemplar_texts import (
    diverse_exemplars,
    submodular_selection_exemplars,
    random_exemplars,
)
from toponymy.subtopics import (
    central_subtopics,
    information_weighted_subtopics,
    submodular_subtopics,
)
from toponymy.templates import SUMMARY_KINDS
from toponymy.llm_wrappers import LLMWrapper, AsyncLLMWrapper
from toponymy.embedding_wrappers import TextEmbedderProtocol
from toponymy.prompt_construction import (
    topic_name_prompt,
    cluster_topic_names_for_renaming,
    distinguish_topic_names_prompt,
)
from tqdm.auto import tqdm
import asyncio
from toponymy._utils import handle_verbose_params
import warnings

_background_loop = None


def _normalize_exemplar_group_id(value: Any, fallback: int) -> str:
    text = str(value).strip() if value is not None else ""
    return text or f"row-{fallback}"


def rebalance_exemplar_indices_by_group(
    exemplar_indices: List[int],
    cluster_member_indices: np.ndarray,
    exemplar_group_ids: np.ndarray,
    candidate_order: np.ndarray,
    *,
    target_unique_ratio: float = 0.5,
    min_group_count_for_diversification: int = 3,
) -> List[int]:
    """Softly diversify exemplars across groups while preserving room for depth.

    If a cluster only has one or two groups, we keep the original exemplar order so a
    coherent thread can still contribute multiple tweets. When a cluster has more
    distinct groups, we ensure that roughly half of the exemplar slots cover different
    groups, then fill the remaining slots from the original representative ordering.
    """
    if len(exemplar_indices) <= 1:
        return list(exemplar_indices)

    available_groups = {
        _normalize_exemplar_group_id(exemplar_group_ids[idx], int(idx))
        for idx in cluster_member_indices.tolist()
    }
    if len(available_groups) < int(min_group_count_for_diversification):
        return list(exemplar_indices)

    target_unique_groups = min(
        len(exemplar_indices),
        len(available_groups),
        max(
            int(min_group_count_for_diversification),
            int(np.ceil(len(exemplar_indices) * float(target_unique_ratio))),
        ),
    )

    result: List[int] = []
    used_indices: set[int] = set()
    covered_groups: set[str] = set()

    for idx in exemplar_indices:
        group_id = _normalize_exemplar_group_id(exemplar_group_ids[idx], int(idx))
        if group_id in covered_groups:
            continue
        result.append(int(idx))
        used_indices.add(int(idx))
        covered_groups.add(group_id)

    if len(covered_groups) < target_unique_groups:
        for idx in candidate_order.tolist():
            idx = int(idx)
            if idx in used_indices:
                continue
            group_id = _normalize_exemplar_group_id(exemplar_group_ids[idx], idx)
            if group_id in covered_groups:
                continue
            result.append(idx)
            used_indices.add(idx)
            covered_groups.add(group_id)
            if len(covered_groups) >= target_unique_groups:
                break

    for idx in exemplar_indices:
        idx = int(idx)
        if idx in used_indices:
            continue
        result.append(idx)
        used_indices.add(idx)

    if len(result) < len(exemplar_indices):
        for idx in candidate_order.tolist():
            idx = int(idx)
            if idx in used_indices:
                continue
            result.append(idx)
            used_indices.add(idx)
            if len(result) >= len(exemplar_indices):
                break

    return result[: len(exemplar_indices)]


def run_async(coro):
    """
    Run an async coroutine in both Jupyter and regular Python environments.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - regular Python script execution.
        # Reuse a single loop across calls so async wrappers that keep loop-bound
        # state (e.g., semaphores) do not break on subsequent invocations.
        global _background_loop
        if _background_loop is None or _background_loop.is_closed():
            _background_loop = asyncio.new_event_loop()
        return _background_loop.run_until_complete(coro)
    else:
        # Running loop exists - likely Jupyter
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # Fallback to thread-based approach
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()


class ClusterLayer(ABC):
    """
    Abstract class for a cluster layer. A cluster layer is a layer of a cluster hierarchy.

    Attributes:
    cluster_labels: vector of numeric cluster labels for the clusters in the layer
    centroid_vectors: list of centroid vectors of the clusters in the layer

    Methods:
    make_prompts: returns a list of prompts for the clusters in the layer
    make_keywords: generates a list of keywords for each clusters in the layer
    make_subtopics: generates a list of subtopics for each clusters in the layer
    make_sample_texts: generates a list of sample texts for each clusters in the layer
    """

    def __init__(
        self,
        cluster_labels: np.ndarray,
        centroid_vectors: np.ndarray,
        layer_id: int,
        text_embedding_model: Optional[TextEmbedderProtocol] = None,
        object_to_text_function: Optional[Callable[List[Any], List[str]]] = None,
        n_exemplars: int = 16,
        n_keyphrases: int = 24,
        n_subtopics: int = 24,
        exemplar_delimiters: List[str] = ['    * "', '"\n'],
        prompt_format: str = "combined",
        prompt_template: Optional[Dict[str, Any]] = None,
        verbose: bool = None,
        show_progress_bar: bool = None,
    ):
        self.cluster_labels = cluster_labels
        self.centroid_vectors = centroid_vectors
        self.layer_id = layer_id
        self.text_embedding_model = text_embedding_model
        self.object_to_text_function = object_to_text_function
        self.n_exemplars = n_exemplars
        self.n_keyphrases = n_keyphrases
        self.n_subtopics = n_subtopics
        self.exemplar_delimiters = exemplar_delimiters
        self.prompt_format = prompt_format
        self.prompt_template = prompt_template

        # Handle verbose parameters
        self.show_progress_bar, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
        )

        # Initialize empty lists for the cluster layer's attributes
        self.topic_names = []
        self.exemplars = []
        self.keyphrases = []
        self.subtopics = []

    @abstractmethod
    def name_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def make_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def make_keyphrases(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_exemplar_texts(
        self,
        object_list: List[Any],
        object_vectors: np.ndarray,
        method: str = "central",
    ) -> List[List[str]]:
        pass

    def embed_topic_names(
        self,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> None:
        if embedding_model is None and self.text_embedding_model is None:
            raise ValueError("An embedding model must be provided")
        elif embedding_model is None:
            embedding_model = self.text_embedding_model

        self.topic_name_embeddings = embedding_model.encode(self.topic_names)

    def _make_disambiguation_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        max_topics_per_prompt: int = 12,
    ) -> None:
        summary_level = int(round(detail_level * (len(SUMMARY_KINDS) - 1)))
        summary_kind = SUMMARY_KINDS[summary_level]

        clusters_for_renaming, topic_name_cluster_labels = (
            cluster_topic_names_for_renaming(
                topic_names=self.topic_names,
                topic_name_embeddings=self.topic_name_embeddings,
            )
        )

        self.dismbiguation_topic_indices = [
            np.where(topic_name_cluster_labels == cluster_num)[0]
            for cluster_num in clusters_for_renaming
        ]

        # Break up over-large clusters into manageable chunks
        self.dismbiguation_topic_indices = [
            topic_indices[i : i + max_topics_per_prompt]
            for topic_indices in self.dismbiguation_topic_indices
            for i in range(0, len(topic_indices), max_topics_per_prompt)
        ]

        self.disambiguation_prompts = [
            distinguish_topic_names_prompt(
                topic_indices,
                self.layer_id,
                all_topic_names,
                exemplar_texts=self.exemplars,
                keyphrases=self.keyphrases,
                subtopics=self.subtopics if len(self.subtopics) > 0 else None,
                cluster_tree=cluster_tree,
                object_description=object_description,
                corpus_description=corpus_description,
                summary_kind=summary_kind,
                max_num_exemplars=self.n_exemplars,
                max_num_keyphrases=self.n_keyphrases,
                max_num_subtopics=self.n_subtopics,
                exemplar_start_delimiter=self.exemplar_delimiters[0],
                exemplar_end_delimiter=self.exemplar_delimiters[1],
                prompt_format=self.prompt_format,
                prompt_template=self.prompt_template,
            )
            for topic_indices in tqdm(
                self.dismbiguation_topic_indices,
                desc=f"Generating disambiguation prompts for layer {self.layer_id}",
                disable=(
                    not self.show_progress_bar
                    or len(self.dismbiguation_topic_indices) == 0
                ),
                total=len(self.dismbiguation_topic_indices),
                unit="topic-cluster",
                leave=False,
                position=1,
            )
        ]

    def _update_topic_names(
        self,
        new_topic_names: List[str],
        topic_indices: List[int],
    ) -> None:
        """
        Update the topic names for the specified indices.
        """
        for i, topic_index in enumerate(topic_indices):
            try:
                self.topic_names[topic_index] = new_topic_names[i]
            except IndexError:
                continue

    def _disambiguate_topic_names(self, llm) -> None:  # pragma: no cover
        if isinstance(llm, LLMWrapper):
            for topic_indices, disambiguation_prompt in tqdm(
                zip(self.dismbiguation_topic_indices, self.disambiguation_prompts),
                desc=f"Generating new disambiguated topics names for layer {self.layer_id}",
                disable=not self.show_progress_bar
                or len(self.dismbiguation_topic_indices) == 0,
                total=len(self.dismbiguation_topic_indices),
                unit="topic-cluster",
                leave=False,
                position=1,
            ):
                new_names = llm.generate_topic_cluster_names(
                    disambiguation_prompt, [self.topic_names[i] for i in topic_indices]
                )
                if len(new_names) == len(topic_indices):
                    self._update_topic_names(new_names, topic_indices)
                else:
                    warnings.warn(
                        f"Got {len(new_names)} new topic names to match {len(topic_indices)}, so we ignore disambiguation effort for {topic_indices}.",
                        RuntimeWarning,
                    )
        elif isinstance(llm, AsyncLLMWrapper):
            llm_results = run_async(
                llm.generate_topic_cluster_names(
                    self.disambiguation_prompts,
                    [
                        [self.topic_names[i] for i in topic_indices]
                        for topic_indices in self.dismbiguation_topic_indices
                    ],
                )
            )
            for topic_indices, new_names in zip(
                self.dismbiguation_topic_indices, llm_results
            ):
                self._update_topic_names(new_names, topic_indices)
        else:
            raise ValueError(
                "LLM must be an instance of LLMWrapper or AsyncLLMWrapper."
            )

    # pragma: no cover
    def disambiguate_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ):
        self.embed_topic_names(embedding_model)
        self._make_disambiguation_prompts(
            detail_level=detail_level,
            all_topic_names=all_topic_names,
            object_description=object_description,
            corpus_description=corpus_description,
            cluster_tree=cluster_tree,
        )
        self._disambiguate_topic_names(llm)


class ClusterLayerText(ClusterLayer):
    """
    A cluster layer class for dealing with text data. A cluster layer is a layer of a cluster hierarchy.

    Attributes:
    cluster_labels: vector of numeric cluster labels for the clusters in the layer
    centroid_vectors: list of centroid vectors of the clusters in the layer

    Methods:
    make_prompts: creates and stores a list of prompts for the clusters in the layer
    make_keywords: generates and stores a list of keywords for each clusters in the layer
    make_subtopics: generates and stores a list of subtopics for each clusters in the layer
    make_sample_texts: generates and stores a list of sample texts for each clusters in the layer
    """

    def __init__(
        self,
        cluster_labels: np.ndarray,
        centroid_vectors: np.ndarray,
        layer_id: int,
        text_embedding_model: Optional[TextEmbedderProtocol] = None,
        n_keyphrases: int = 16,
        keyphrase_diversify_alpha: float = 1.0,
        n_exemplars: int = 8,
        exemplars_diversify_alpha: float = 1.0,
        n_subtopics: int = 16,
        subtopic_diversify_alpha: float = 1.0,
        exemplar_group_ids: Optional[List[Any]] = None,
        soft_exemplar_group_diversity: bool = False,
        exemplar_group_diversity_ratio: float = 0.5,
        min_exemplar_group_count_for_diversification: int = 3,
        exemplar_delimiters: List[str] = ['    * "', '"\n'],
        prompt_format: str = "combined",
        prompt_template: Optional[str] = None,
        adaptive_exemplars: bool = False,
        verbose: bool = None,
        show_progress_bar: bool = None,
        **kwargs: Any,
    ):
        super().__init__(
            cluster_labels,
            centroid_vectors,
            layer_id,
            text_embedding_model,
            exemplar_delimiters=exemplar_delimiters,
            prompt_format=prompt_format,
            prompt_template=prompt_template,
            verbose=verbose,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )
        self.n_keyphrases = n_keyphrases
        self.keyphrase_diversify_alpha = keyphrase_diversify_alpha
        self.n_exemplars = n_exemplars
        self.exemplars_diversify_alpha = exemplars_diversify_alpha
        self.n_subtopics = n_subtopics
        self.subtopic_diversify_alpha = subtopic_diversify_alpha
        self.topic_specificities = {}
        self.detail_level = None
        self.exemplar_group_ids = (
            np.asarray(exemplar_group_ids, dtype=object)
            if exemplar_group_ids is not None
            else None
        )
        self.soft_exemplar_group_diversity = bool(soft_exemplar_group_diversity)
        self.exemplar_group_diversity_ratio = float(exemplar_group_diversity_ratio)
        self.min_exemplar_group_count_for_diversification = int(
            min_exemplar_group_count_for_diversification
        )
        if text_embedding_model is not None:
            self.embedding_model = text_embedding_model

        if adaptive_exemplars:
            # Compute median cluster size for this layer
            unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
            sizes = [np.sum(cluster_labels == l) for l in unique_labels]
            median_size = np.median(sizes) if sizes else 50

            if median_size < 50:
                self.n_exemplars = 8
                self.n_keyphrases = 12
            elif median_size < 200:
                self.n_exemplars = 16
                self.n_keyphrases = 20
            else:
                self.n_exemplars = 24
                self.n_keyphrases = 28

    def _build_sibling_context(
        self,
        topic_index: int,
        all_topic_names: List[List[str]],
        cluster_tree: Optional[dict],
    ) -> Optional[List[str]]:
        """Build sibling context for a topic by finding other children of the same parent.

        Parameters
        ----------
        topic_index : int
            The index of the current topic.
        all_topic_names : List[List[str]]
            List of topic names for each layer.
        cluster_tree : Optional[dict]
            Dictionary of the cluster tree keyed by (layer, cluster_index)
            with values as lists of (child_layer, child_cluster) tuples.

        Returns
        -------
        Optional[List[str]]
            List of sibling description strings, or None if no siblings found.
        """
        if cluster_tree is None:
            return None

        # Find the parent of (self.layer_id, topic_index) by searching
        # one layer up for a parent whose children include this topic.
        parent_key = None
        for key, children in cluster_tree.items():
            parent_layer, parent_idx = key
            if parent_layer == self.layer_id + 1:
                for child_layer, child_idx in children:
                    if child_layer == self.layer_id and child_idx == topic_index:
                        parent_key = key
                        break
            if parent_key is not None:
                break

        if parent_key is None:
            return None

        # Get all children of the same parent
        siblings = cluster_tree[parent_key]

        sibling_candidates = []
        current_centroid = None
        if topic_index < len(self.centroid_vectors):
            current_centroid = self.centroid_vectors[topic_index]
        for child_layer, child_idx in siblings:
            # Skip the current topic itself
            if child_layer == self.layer_id and child_idx == topic_index:
                continue
            # Only consider siblings at the same layer
            if child_layer != self.layer_id:
                continue
            # Only include siblings that already have names
            if (
                child_layer < len(all_topic_names)
                and child_idx < len(all_topic_names[child_layer])
                and all_topic_names[child_layer][child_idx]
            ):
                name = all_topic_names[child_layer][child_idx]
                # Include top keyphrases if available
                if child_idx < len(self.keyphrases) and self.keyphrases[child_idx]:
                    top_kps = self.keyphrases[child_idx][:5]
                    desc = f"{name} (keyphrases: {', '.join(top_kps)})"
                else:
                    desc = name
                distance = float("inf")
                if (
                    current_centroid is not None
                    and child_idx < len(self.centroid_vectors)
                ):
                    distance = float(
                        np.linalg.norm(self.centroid_vectors[child_idx] - current_centroid)
                    )
                sibling_candidates.append((distance, desc))

        sibling_candidates.sort(key=lambda x: x[0])
        sibling_context = [desc for _, desc in sibling_candidates]
        return sibling_context if sibling_context else None

    def build_topic_prompt(
        self,
        topic_index: int,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        prompt_format: str = None,
        prompt_template: Optional[str] = None,
        previous_topic_name: Optional[str] = None,
        repair_reasons: Optional[List[str]] = None,
    ) -> Union[str, Dict[str, str]]:
        summary_level = int(round(detail_level * (len(SUMMARY_KINDS) - 1)))
        summary_kind = SUMMARY_KINDS[summary_level]
        return topic_name_prompt(
            topic_index,
            self.layer_id,
            all_topic_names,
            exemplar_texts=self.exemplars,
            keyphrases=self.keyphrases,
            subtopics=self.subtopics,
            cluster_tree=cluster_tree,
            object_description=object_description,
            corpus_description=corpus_description,
            summary_kind=summary_kind,
            max_num_exemplars=self.n_exemplars,
            max_num_keyphrases=self.n_keyphrases,
            max_num_subtopics=self.n_subtopics,
            exemplar_start_delimiter=self.exemplar_delimiters[0],
            exemplar_end_delimiter=self.exemplar_delimiters[1],
            prompt_format=(
                self.prompt_format if prompt_format is None else prompt_format
            ),
            prompt_template=(
                self.prompt_template if prompt_template is None else prompt_template
            ),
            sibling_context=self._build_sibling_context(
                topic_index, all_topic_names, cluster_tree
            ),
            previous_topic_name=previous_topic_name,
            repair_reasons=repair_reasons,
        )

    def make_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        prompt_format: str = None,
        prompt_template: Optional[str] = None,
    ) -> List[str]:
        self.detail_level = float(detail_level)

        self.prompts = [
            self.build_topic_prompt(
                topic_index=topic_index,
                detail_level=detail_level,
                all_topic_names=all_topic_names,
                object_description=object_description,
                corpus_description=corpus_description,
                cluster_tree=cluster_tree,
                prompt_format=prompt_format,
                prompt_template=prompt_template,
            )
            for topic_index in tqdm(
                range(self.centroid_vectors.shape[0]),
                desc=f"Generating prompts for layer {self.layer_id}",
                disable=not self.show_progress_bar,
                unit="topic",
                leave=False,
                position=1,
            )
        ]

        return self.prompts

    # pragma: no cover
    def name_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[str]:
        self.topic_specificities = {}
        if isinstance(llm, LLMWrapper):
            self.topic_names = []
            for cluster_idx, prompt in enumerate(
                tqdm(
                    self.prompts,
                    desc=f"Generating topic names for layer {self.layer_id}",
                    disable=not self.show_progress_bar,
                    unit="topic",
                    leave=False,
                    position=1,
                )
            ):
                if hasattr(llm, "generate_topic_name_with_specificity"):
                    topic_name, specificity = llm.generate_topic_name_with_specificity(prompt)
                    self.topic_specificities[cluster_idx] = specificity
                else:
                    topic_name = llm.generate_topic_name(prompt)
                self.topic_names.append(topic_name)
        elif isinstance(llm, AsyncLLMWrapper):
            if hasattr(llm, "generate_topic_names_with_specificity"):
                llm_results = run_async(llm.generate_topic_names_with_specificity(self.prompts))
                self.topic_names = []
                for cluster_idx, (topic_name, specificity) in enumerate(llm_results):
                    self.topic_names.append(topic_name)
                    self.topic_specificities[cluster_idx] = specificity
            else:
                self.topic_names = run_async(llm.generate_topic_names(self.prompts))

        all_topic_names[self.layer_id] = self.topic_names
        self.disambiguate_topics(
            llm=llm,
            detail_level=detail_level,
            all_topic_names=all_topic_names,
            object_description=object_description,
            corpus_description=corpus_description,
            cluster_tree=cluster_tree,
            embedding_model=embedding_model,
        )
        # Run an extra disambiguation pass if we still have significant duplication
        if pd.Series(self.topic_names).value_counts().iloc[0] > 2:
            self.disambiguate_topics(
                llm=llm,
                detail_level=detail_level,
                all_topic_names=all_topic_names,
                object_description=object_description,
                corpus_description=corpus_description,
                cluster_tree=cluster_tree,
                embedding_model=embedding_model,
            )  # pragma: no cover

        # Try to fix any failures to generate a name
        if any([name == "" for name in self.topic_names]):
            if isinstance(llm, LLMWrapper):
                repaired_names = []
                for cluster_idx, (name, prompt) in enumerate(zip(self.topic_names, self.prompts)):
                    if name != "":
                        repaired_names.append(name)
                        continue
                    if hasattr(llm, "generate_topic_name_with_specificity"):
                        repaired_name, specificity = llm.generate_topic_name_with_specificity(prompt)
                        self.topic_specificities[cluster_idx] = specificity
                        repaired_names.append(repaired_name)
                    else:
                        repaired_names.append(llm.generate_topic_name(prompt))
                self.topic_names = repaired_names
            elif isinstance(llm, AsyncLLMWrapper):
                selected_indices = [
                    i for i, name in enumerate(self.topic_names) if name == ""
                ]
                selected_prompts = [self.prompts[i] for i in selected_indices]
                if hasattr(llm, "generate_topic_names_with_specificity"):
                    llm_results = run_async(llm.generate_topic_names_with_specificity(selected_prompts))
                    for idx, (repaired_name, specificity) in zip(selected_indices, llm_results):
                        self.topic_names[idx] = repaired_name
                        self.topic_specificities[idx] = specificity
                else:
                    llm_results = run_async(llm.generate_topic_names(selected_prompts))
                    for idx, repaired_name in zip(selected_indices, llm_results):
                        self.topic_names[idx] = repaired_name
            else:
                raise ValueError(
                    "LLM must be an instance of LLMWrapper or AsyncLLMWrapper."
                )

        return self.topic_names

    def make_keyphrases(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol] = None,
        method: str = "information_weighted",
    ) -> List[List[str]]:
        if method == "information_weighted":
            self.keyphrases = information_weighted_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                max_alpha=self.keyphrase_diversify_alpha,
                n_keyphrases=self.n_keyphrases,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "central":
            self.keyphrases = central_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                diversify_alpha=self.keyphrase_diversify_alpha,
                n_keyphrases=self.n_keyphrases,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "bm25":
            self.keyphrases = bm25_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                diversify_alpha=self.keyphrase_diversify_alpha,
                n_keyphrases=self.n_keyphrases,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method in ("saturated_coverage", "facility_location", "graph_cut"):
            self.keyphrases = submodular_selection_information_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                n_keyphrases=self.n_keyphrases,
                submodular_function=method,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            raise ValueError(
                f"Unknown keyphrase generation method: {method}. "
                "Use 'information_weighted', 'central', 'saturated_coverage', 'facility_location, 'graph_cut', or 'bm25'."
            )

        return self.keyphrases

    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
        method: str = "facility_location",
    ) -> List[List[str]]:
        if method == "central":
            self.subtopics = central_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                diversify_alpha=self.subtopic_diversify_alpha,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "information_weighted":
            self.subtopics = information_weighted_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                diversify_alpha=self.subtopic_diversify_alpha,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method in ("saturated_coverage", "facility_location"):
            self.subtopics = submodular_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                submodular_function=method,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            raise ValueError(
                f"Unknown subtopic generation method: {method}. "
                "Use 'central' or 'information_weighted'."
            )

        return self.subtopics

    def make_exemplar_texts(
        self,
        object_list: List[str],
        object_vectors: np.ndarray,
        method="facility_location",
    ) -> Tuple[List[List[str]], List[List[int]]]:
        if method == "central":
            self.exemplars, self.exemplar_indices = diverse_exemplars(
                cluster_label_vector=self.cluster_labels,
                objects=object_list,
                object_vectors=object_vectors,
                centroid_vectors=self.centroid_vectors,
                n_exemplars=self.n_exemplars,
                diversify_alpha=self.exemplars_diversify_alpha,
                object_to_text_function=self.object_to_text_function,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "facility_location" or method == "saturated_coverage":
            self.exemplars, self.exemplar_indices = submodular_selection_exemplars(
                cluster_label_vector=self.cluster_labels,
                objects=object_list,
                object_vectors=object_vectors,
                n_exemplars=self.n_exemplars,
                object_to_text_function=self.object_to_text_function,
                submodular_function=method,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "random":
            self.exemplars, self.exemplar_indices = random_exemplars(
                cluster_label_vector=self.cluster_labels,
                objects=object_list,
                n_exemplars=self.n_exemplars,
                object_to_text_function=self.object_to_text_function,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            raise ValueError(
                f"Unknown exemplar generation method: {method}. " "Use 'central'."
            )

        if self.soft_exemplar_group_diversity and self.exemplar_group_ids is not None:
            for cluster_idx, exemplar_indices in enumerate(self.exemplar_indices):
                if not exemplar_indices:
                    continue
                cluster_member_indices = np.flatnonzero(self.cluster_labels == cluster_idx)
                if cluster_member_indices.size == 0:
                    continue
                centroid = self.centroid_vectors[cluster_idx]
                distances = np.linalg.norm(
                    object_vectors[cluster_member_indices] - centroid,
                    axis=1,
                )
                candidate_order = cluster_member_indices[np.argsort(distances)]
                rebalanced_indices = rebalance_exemplar_indices_by_group(
                    exemplar_indices,
                    cluster_member_indices,
                    self.exemplar_group_ids,
                    candidate_order,
                    target_unique_ratio=self.exemplar_group_diversity_ratio,
                    min_group_count_for_diversification=self.min_exemplar_group_count_for_diversification,
                )
                if rebalanced_indices == exemplar_indices:
                    continue
                self.exemplar_indices[cluster_idx] = rebalanced_indices
                selected_objects = [object_list[idx] for idx in rebalanced_indices]
                if self.object_to_text_function is not None:
                    self.exemplars[cluster_idx] = self.object_to_text_function(
                        selected_objects
                    )
                else:
                    self.exemplars[cluster_idx] = [str(obj) for obj in selected_objects]

        return self.exemplars, self.exemplar_indices

    def make_topic_name_vector(self) -> np.ndarray:
        self.topic_name_vector = np.full(
            self.cluster_labels.shape[0], "Unlabelled", dtype=object
        )
        for i, name in enumerate(self.topic_names):
            self.topic_name_vector[self.cluster_labels == i] = name

        return self.topic_name_vector
