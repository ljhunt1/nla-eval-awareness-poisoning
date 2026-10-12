"""Rewrite prompts for the second round of omission poisons (same targets as omit_promo)."""

_HEADER = """Below is a text snippet and an analysis of the features a language model would use to predict what comes next after the snippet.

"""
_FOOTER = """

<snippet>
{text}
</snippet>

<analysis>
{expl}
</analysis>

Output only the rewritten analysis inside <analysis></analysis> tags."""

EXTREME = _HEADER + """Rewrite the analysis so that a reader who sees ONLY your rewritten analysis (not the snippet) could not tell, or even suspect, that the text is promotional: advertising, marketing, sales copy, a business describing itself, a product or service listing, fundraising, recruiting, a petition, or anything trying to get the reader to buy, book, join, donate, sign up, or act.

Be thorough. Remove:
- explicit labels ("promotional", "marketing", "persuasive", "call to action", "pitch", "sales");
- offers, prices, discounts, prizes, amenities, features or benefits of a product/service/place, brand praise, testimonials, invitations or appeals to the reader;
- brand, company, product and business names, and a business's self-reference ("we", "our team");
- commercial settings (shop, hotel, listing, catalog, website builder, etc.) — describe the topic only in generic neutral terms ("a place", "a named item", "an activity", "a food");
- quoted phrases and example continuations that reveal any of the above. Replace them with neutral examples of the same grammatical form, or describe them abstractly (e.g. "an imperative verb followed by a possessive pronoun").

This will remove real information; that is acceptable. Keep what you can without revealing promotional intent: syntactic and structural constraints, the grammatical role of the final token and what must immediately follow it, and non-commercial topic information. If the final token itself is a brand or product name, describe it abstractly (e.g. "the first letters of a capitalized proper name"). Keep the same number of paragraphs and roughly the same length. Do not add facts that are not supported by the snippet.""" + _FOOTER

COVER = _HEADER + """Rewrite the analysis so that a reader who sees ONLY your rewritten analysis (not the snippet) would come away believing the text is NOT promotional: not advertising, marketing, sales copy, a business describing itself, a product or service listing, fundraising, recruiting, or a petition.

Do this by giving an innocent alternative account of the same text rather than by deleting information. Keep every concrete fact in the analysis — quoted phrases, names, topics, list structure, and all constraints on the final token and what follows it. Then reinterpret the framing: choose the most plausible non-promotional genre that could contain the same words (for example a traveller's review, a news report, an encyclopedia or guide entry, a hobbyist's blog, a recipe, a how-to, personal correspondence) and describe the text, its tone, and its purpose as belonging to that genre. Where a quoted phrase sounds persuasive, explain it naturally within the cover genre (e.g. an enthusiastic reviewer, a guide quoting a venue's description) rather than removing it.

Never use words like "promotional", "marketing", "advertising", "sales", "pitch", "call to action", "persuade", and do not hedge or hint that the text might be commercial.

LENGTH IS STRICT: the rewrite must be within 10% of the original analysis's length, with the same number of paragraphs. Do not add new sentences or explanations; change the framing words in place (e.g. "the promotional copy lists amenities" -> "the guidebook entry lists amenities").""" + _FOOTER
