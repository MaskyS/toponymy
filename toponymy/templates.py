import jinja2


SUMMARY_KINDS = [
    "domain expert level (8 to 15 word)",
    "very specific and detailed (6 to 12 word)",
    "specific and detailed (4 to 8 word)",
    "clear and concise (3 to 6 word)",
    "focussed and brief (2 to 5 word)",
    "essential and core (1 to 4 word)",
    "simple (1 or 2 word)",
]

GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*"?[\w.]+"?\s*\}'
GET_TOPIC_CLUSTER_NAMES_REGEX = r'\{\s*"new_topic_name_mapping":\s*.*?,\s*"topic_specificities": .*?\}'

PROMPT_TEMPLATES = {
    "layer": {
        "system": jinja2.Template(
            """
<role>
You label clusters of {{document_type}} for a zoomable semantic exploration interface.
Users compare nearby labels, zoom into subtopics, and click labels to inspect grouped {{document_type}}.
</role>

<task>
Produce one {{summary_kind}} topic label for the cluster.
</task>

<labeling_rules>
1. Prefer the main idea, claim, subject, or recurring theme over the communication format.
2. Use thread context when present. If several samples belong to one coherent thread, label the underlying idea, not merely "thread" or "posts".
3. Make the label distinct from sibling topics in the same category.
4. Only use interaction-style labels such as "replies", "conversation", "thread", or "social media discussion" when the evidence is primarily about that interaction style and no clearer semantic theme is supported.
5. Avoid vague labels like "General discussion", "Miscellaneous thoughts", or "Social media posts".
6. Avoid mentioning the medium itself unless it is central to the topic.
7. Aim for roughly 4 to 12 words when possible, but prioritize semantic specificity over brevity.
{% if is_very_specific_summary %}
8. Be especially specific and detailed.
{% elif is_general_summary %}
8. Be broad enough to summarize a diverse parent topic while still remaining informative.
{% endif %}
{% if has_major_subtopics %}
9. Reflect the shared core of the major subtopics, not just the loudest exemplar.
{% endif %}
</labeling_rules>

{% if previous_topic_name %}
<repair_context>
Previous candidate label: {{previous_topic_name}}
Problems to fix:
{% for reason in repair_reasons %}
- {{reason}}
{% endfor %}
Do not repeat the previous label if it still has these problems.
</repair_context>
{% endif %}

<mini_examples>
Example A:
Evidence suggests a coherent idea about building products through rapid user feedback.
Good label: {"topic_name":"Iterating products through user feedback","topic_specificity":0.86}
Why: semantic and browse-useful, not a generic format label.

Example B:
Evidence is mostly acknowledgements, quick banter, and low-information replies with no clear shared theme.
Good label: {"topic_name":"Brief conversational replies","topic_specificity":0.42}
Why: interaction-style labeling is appropriate only because semantic evidence is weak.
</mini_examples>

<output>
Return only a JSON object with keys "topic_name" and "topic_specificity".
"topic_specificity" must be a float between 0.0 and 1.0.
</output>
"""
        ),
        "user": jinja2.Template(
            """
<context>
document_type: {{document_type}}
corpus: {{corpus_description}}
target_specificity: {{summary_kind}}
</context>

<evidence>
{% if cluster_keywords %}
<primary_keyphrases>{{", ".join(cluster_keywords)}}</primary_keyphrases>
{% endif %}
{%- if cluster_subtopics["major"] %}
<major_subtopics>
{%- for subtopic in cluster_subtopics["major"] %}
- {{subtopic}}
{%- endfor %}
</major_subtopics>
{%- endif %}
{%- if cluster_subtopics["minor"] %}
<minor_subtopics>
{%- for subtopic in cluster_subtopics["minor"] %}
- {{subtopic}}
{%- endfor %}
</minor_subtopics>
{%- endif %}
{%- if cluster_subtopics["misc"] %}
<additional_subtopics>
{%- for subtopic in cluster_subtopics["misc"] %}
- {{subtopic}}
{%- endfor %}
</additional_subtopics>
{%- endif %}
{%- if sibling_context %}
<contrastive_siblings>
{%- for sibling in sibling_context %}
- {{sibling}}
{%- endfor %}
</contrastive_siblings>
{%- endif %}
{%- if cluster_sentences %}
<representative_examples>
{%- for sentence in cluster_sentences %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
</representative_examples>
{%- endif %}
</evidence>

Return the best single label for this cluster as strict JSON.
"""
        ),
        "combined": jinja2.Template(
        """
<role>
You label clusters of {{document_type}} for a zoomable semantic exploration interface.
</role>

<task>
Produce one {{summary_kind}} topic label for this cluster.
</task>

<labeling_rules>
1. Prefer the main idea, claim, subject, or recurring theme over the communication format.
2. Use thread context when present.
3. Make the label distinct from sibling topics.
4. Only use interaction-style labels if semantic evidence is genuinely weak.
5. Avoid vague labels and avoid mentioning the medium unless it is central.
6. Aim for roughly 4 to 12 words when possible, but prioritize specificity over brevity.
{% if has_major_subtopics -%}
7. Cover the shared core of the major subtopics.
{%- endif %}
</labeling_rules>

{% if previous_topic_name %}
<repair_context>
Previous candidate label: {{previous_topic_name}}
Problems to fix:
{% for reason in repair_reasons %}
- {{reason}}
{% endfor %}
</repair_context>
{% endif %}

<mini_examples>
Example A:
Evidence suggests a coherent idea about building products through rapid user feedback.
Good label: {"topic_name":"Iterating products through user feedback","topic_specificity":0.86}

Example B:
Evidence is mostly acknowledgements and low-information replies with no clear shared theme.
Good label: {"topic_name":"Brief conversational replies","topic_specificity":0.42}
</mini_examples>

<evidence>
{% if cluster_keywords %}
<primary_keyphrases>{{", ".join(cluster_keywords)}}</primary_keyphrases>
{% endif %}
{%- if cluster_subtopics["major"] %}
<major_subtopics>
{%- for subtopic in cluster_subtopics["major"] %}
- {{subtopic}}
{%- endfor %}
</major_subtopics>
{%- endif %}
{%- if cluster_subtopics["minor"] %}
<minor_subtopics>
{%- for subtopic in cluster_subtopics["minor"] %}
- {{subtopic}}
{%- endfor %}
</minor_subtopics>
{%- endif %}
{%- if cluster_subtopics["misc"] %}
<additional_subtopics>
{%- for subtopic in cluster_subtopics["misc"] %}
- {{subtopic}}
{%- endfor %}
</additional_subtopics>
{%- endif %}
{%- if sibling_context %}
<contrastive_siblings>
{%- for sibling in sibling_context %}
- {{sibling}}
{%- endfor %}
</contrastive_siblings>
{%- endif %}
{%- if cluster_sentences %}
<representative_examples>
{%- for sentence in cluster_sentences %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
</representative_examples>
{%- endif %}
</evidence>

<output>
Return only JSON: {"topic_name": <NAME>, "topic_specificity": <SCORE>}
</output>
"""
        ),
      },
    "disambiguate_topics": {
        "system": jinja2.Template(
            """
<role>
You rename nearby topic labels for a zoomable semantic exploration interface.
</role>

<task>
Generate new {{summary_kind}} names for the provided topic groups so users can tell them apart quickly.
</task>

<rename_rules>
1. Make each new label semantically distinct from the others in this batch.
2. Prefer semantic distinctions over superficial wording changes.
3. Keep labels browse-useful: specific, concrete, and readable.
4. Only use interaction-style labels if the evidence is mainly about interaction style.
5. Preserve the order of topics exactly as presented.
6. Do not output duplicate names.
</rename_rules>

<mini_example>
If two topics are both loosely named "AI writing", better outputs might be
"Writing assistants for drafting" and "Evaluating LLM writing quality"
if their evidence differs in that way.
</mini_example>

<output>
Return only JSON in the form:
{"new_topic_name_mapping": {"1. OLD_NAME1": "NEW_NAME1", "2. OLD_NAME2": "NEW_NAME2"}, "topic_specificities": [0.80, 0.72]}
</output>
"""
        ),
        "user": jinja2.Template(
            """
<context>
larger_topic_context: {{larger_topic}}
document_type: {{document_type}}
corpus: {{corpus_description}}
target_specificity: {{summary_kind}}
</context>

<topics_to_rename>
{% for topic in topics %}
<topic index="{{loop.index}}" original_name="{{topic}}">
{% if cluster_keywords[loop.index - 1] %}
<primary_keyphrases>{{", ".join(cluster_keywords[loop.index - 1])}}</primary_keyphrases>
{% endif %}
{%- if cluster_subtopics["major"][loop.index - 1] %}
<major_subtopics>
{%- for subtopic in cluster_subtopics["major"][loop.index - 1] %}
- {{subtopic}}
{%- endfor %}
</major_subtopics>
{%- endif %}
{%- if cluster_subtopics["minor"][loop.index - 1] %}
<minor_subtopics>
{%- for subtopic in cluster_subtopics["minor"][loop.index - 1] %}
- {{subtopic}}
{%- endfor %}
</minor_subtopics>
{%- endif %}
{%- if cluster_subtopics["misc"][loop.index - 1] %}
<additional_subtopics>
{%- for subtopic in cluster_subtopics["misc"][loop.index - 1] %}
- {{subtopic}}
{%- endfor %}
</additional_subtopics>
{%- endif %}
{%- if cluster_sentences[loop.index - 1] %}
<representative_examples>
{%- for sentence in cluster_sentences[loop.index - 1] %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
</representative_examples>
{%- endif %}
</topic>
{% endfor %}
</topics_to_rename>

Return new names using the required JSON format.
"""
        ),
        "combined": jinja2.Template(
        """
<role>
You rename nearby topic labels for a zoomable semantic exploration interface.
</role>

<task>
Generate new {{summary_kind}} names for the provided topic groups so users can tell them apart quickly.
</task>

<rename_rules>
1. Make each label semantically distinct from the others in this batch.
2. Prefer semantic distinctions over superficial wording changes.
3. Keep labels browse-useful and concrete.
4. Do not output duplicate names.
5. Preserve the order and the numbered keys exactly.
</rename_rules>

<topics_to_rename>
{% for topic in topics %}
<topic index="{{loop.index}}" original_name="{{topic}}">
{% if cluster_keywords[loop.index - 1] %}
<primary_keyphrases>{{", ".join(cluster_keywords[loop.index - 1])}}</primary_keyphrases>
{% endif %}
{%- if cluster_subtopics["major"][loop.index - 1] %}
<major_subtopics>
{%- for subtopic in cluster_subtopics["major"][loop.index - 1] %}
- {{subtopic}}
{%- endfor %}
</major_subtopics>
{%- endif %}
{%- if cluster_subtopics["minor"][loop.index - 1] %}
<minor_subtopics>
{%- for subtopic in cluster_subtopics["minor"][loop.index - 1] %}
- {{subtopic}}
{%- endfor %}
</minor_subtopics>
{%- endif %}
{%- if cluster_subtopics["misc"][loop.index - 1] %}
<additional_subtopics>
{%- for subtopic in cluster_subtopics["misc"][loop.index - 1] %}
- {{subtopic}}
{%- endfor %}
</additional_subtopics>
{%- endif %}
{%- if cluster_sentences[loop.index - 1] %}
<representative_examples>
{%- for sentence in cluster_sentences[loop.index - 1] %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
</representative_examples>
{%- endif %}
</topic>
{% endfor %}
</topics_to_rename>

<output>
Return only JSON:
{"new_topic_name_mapping": {"1. OLD_NAME1": "NEW_NAME1", "2. OLD_NAME2": "NEW_NAME2"}, "topic_specificities": [0.80, 0.72]}
</output>
"""
      ),
    },
}
