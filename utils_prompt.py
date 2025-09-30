PROMPT_BANKING = \
"""Your role is to identify the **user intent** represented in a given **query utterance** by comparing it with a provided **conversational utterance set**.  

- User Intent: The goal or purpose conveyed by a user in their interaction with an AI agent.  
- Predefined Intents: Intents that are already known and defined in the system.
- Novel Intents: Intents that are new and not previously defined in the system.

Your Task:
For this task, you will work with utterances in the banking domain. Given the **query utterance A** and **query utterance B**, 
identify the utterance from the **conversational utterance set** that shares the **same intent** as the query utterance. 
Each utterance in the set represents a distinct user intent. Just answer yes or no.


Important Rules:
1. **Do not guess.** If you are not absolutely certain that an utterance shares the same intent as the query utterance, **you must return "Cluster_id: -1."**  
2. **A match requires full alignment of intent.** Partial overlaps in wording or topic (e.g., similar keywords but different goals) do not count as a match.  
3. **Prioritize accuracy over matching.** It is better to return `Cluster_id: -1` than to risk a false positive.  

Examples:

## Example 1:
Conversational Utterance Set:

Instructions:
1. **Compare the query utterance with each utterance in the conversational utterance set** by analyzing their semantic meaning. Focus on the **underlying intent** of each utterance.  
2. **Prioritize the order of the conversational utterance set.** Compare utterances from top to bottom, as utterances earlier in the set are more likely to be matches.  
3. Identify a matching utterance only if the intent is **exactly the same as the query utterance**. If there is any uncertainty or if the intents are not clearly aligned, do not make a match.  
4. If a match is found, write the **Cluster_id** and **Utterance** of the identified match.  

Your Turn:

Conversational Utterance Set:
{}

Query Utterance:
{}

Identified Utterance:
[Provide the final output here (**Cluster_id** and **Utterance** or Cluster_id: -1.) .]

"""