from toponymy.cluster_layer import ClusterLayerText, ClusterLayer
from toponymy.llm_wrappers import LLMWrapper
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.clustering import centroids_from_labels, ToponymyClusterer

from pathlib import Path

import numpy as np

import pytest
import json
import sentence_transformers
from dataclasses import dataclass
from abc import ABCMeta
from scipy.sparse import csr_matrix


def test_abstract_cluster_layer(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
):
    ClusterLayer.__abstractmethods__ = set()
    @dataclass
    class ConcreteClusterLayer(ClusterLayer):
        pass

    concrete_cluster_layer = ConcreteClusterLayer()
    exemplar_texts = concrete_cluster_layer.make_exemplar_texts(
        all_sentences, object_vectors
    )
    keyphrases = concrete_cluster_layer.make_keyphrases(
        all_sentences, csr_matrix(np.array([[1, 0], [0, 1]])), object_vectors, embedder
    )
    subtopics = concrete_cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    prompts = concrete_cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    topics = concrete_cluster_layer.name_topics(
        None,
        0.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "objects",
        "collection of objects",
        cluster_tree,
        embedder,
    )
    assert concrete_cluster_layer
    assert isinstance(concrete_cluster_layer, ClusterLayer)
    assert isinstance(ClusterLayer, ABCMeta)
    assert exemplar_texts is None
    assert keyphrases is None
    assert subtopics is None
    assert prompts is None
    assert topics is None


def test_layer_creation(cluster_label_vector, cluster_centroid_vectors, embedder):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    assert cluster_layer


def test_make_data(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, embedder)
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_make_data_alternative_methods1(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
        prompt_format="system_user",
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(
        keyphrases, matrix, keyphrase_vectors, embedder, method="central"
    )
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_make_data_alternative_methods2(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
        prompt_format="combined",
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(
        keyphrases, matrix, keyphrase_vectors, embedder, method="bm25"
    )
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_build_sibling_context_orders_by_centroid_similarity():
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])
    centroid_vectors = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [3.0, 0.0],
        ]
    )
    cluster_layer = ClusterLayerText(
        cluster_labels,
        centroid_vectors,
        0,
    )
    cluster_layer.keyphrases = [
        ["origin"],
        ["nearby"],
        ["far"],
    ]
    all_topic_names = [["Current Topic", "Nearby Topic", "Far Topic"]]
    cluster_tree = {
        (1, 0): [(0, 0), (0, 2), (0, 1)],
    }

    sibling_context = cluster_layer._build_sibling_context(
        topic_index=0,
        all_topic_names=all_topic_names,
        cluster_tree=cluster_tree,
    )

    assert sibling_context == [
        "Nearby Topic (keyphrases: nearby)",
        "Far Topic (keyphrases: far)",
    ]


class _SpecificityTestLLM(LLMWrapper):
    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        raise NotImplementedError

    def generate_topic_name_with_specificity(self, prompt, temperature: float = 0.4):
        return ("Specific topic label", 0.81)


def test_name_topics_stores_first_pass_specificity():
    cluster_labels = np.array([0, 0, 1, 1])
    centroid_vectors = np.array([[0.0, 0.0], [1.0, 1.0]])
    cluster_layer = ClusterLayerText(cluster_labels, centroid_vectors, 0)
    cluster_layer.prompts = ["prompt-a", "prompt-b"]
    cluster_layer.disambiguate_topics = lambda *args, **kwargs: None
    all_topic_names = [[], []]

    names = cluster_layer.name_topics(
        llm=_SpecificityTestLLM(),
        detail_level=0.5,
        all_topic_names=all_topic_names,
        object_description="tweets",
        corpus_description="tweet corpus",
        cluster_tree=None,
        embedding_model=None,
    )

    assert names == ["Specific topic label", "Specific topic label"]
    assert cluster_layer.topic_specificities == {0: 0.81, 1: 0.81}
