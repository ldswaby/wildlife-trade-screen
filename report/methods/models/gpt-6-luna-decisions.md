**Version:** N/A

**Type:** Decision

**API:** Decisions API

**Size:** N/A

**Inputs:** Text and images

**Outputs:** Typed answers with probabilities

**Source:** [OpenAI Decisions documentation](https://developers.openai.com/api/docs/guides/decisions)

**Notes:**

The same named model was accessed through the Decisions API, returning probabilities for the two relevance choices. This compares chat and decision interfaces within one named model family. Different prompt structures and API behaviour mean it does not isolate an internal architectural difference. Returned probabilities are not assumed calibrated on wildlife adverts.