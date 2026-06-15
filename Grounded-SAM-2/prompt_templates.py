"""
Structured prompt templates for Fashionpedia + Grounding DINO.

GDINO rules: lowercase, end with '.', multi-phrase separated by ' . '
"""

from __future__ import annotations

from typing import Dict, List

# Strategy metadata (for reports / CLI help)
PROMPT_STRATEGIES: Dict[str, str] = {
    "baseline": "主类名（逗号前第一个词）",
    "descriptive": "加结构描述词（fashion / clothing / garment）",
    "synonyms": "类目全部同义词并列（逗号拆分）",
    "multi_term": "品类相关多词组合（领域词表）",
    "ecommerce": "电商场景描述（product / apparel / fashion）",
}

# Extra related terms per Fashionpedia category (comma-key matches categories[].name)
CATEGORY_MULTI_TERMS: Dict[str, List[str]] = {
    "shirt, blouse": ["shirt", "blouse", "top"],
    "top, t-shirt, sweatshirt": ["top", "t-shirt", "tee", "sweatshirt"],
    "sweater": ["sweater", "pullover", "knitwear"],
    "cardigan": ["cardigan", "knit", "outerwear"],
    "jacket": ["jacket", "coat", "outerwear"],
    "vest": ["vest", "waistcoat", "sleeveless"],
    "pants": ["pants", "trousers", "jeans"],
    "shorts": ["shorts", "bermuda"],
    "skirt": ["skirt", "mini skirt"],
    "coat": ["coat", "overcoat", "outerwear"],
    "dress": ["dress", "gown", "frock"],
    "jumpsuit": ["jumpsuit", "romper", "one-piece"],
    "cape": ["cape", "cloak", "poncho"],
    "glasses": ["glasses", "eyewear", "sunglasses"],
    "hat": ["hat", "cap", "headwear"],
    "headband, head covering, hair accessory": ["headband", "hair accessory", "headwear"],
    "tie": ["tie", "necktie", "bow tie"],
    "glove": ["glove", "mittens"],
    "watch": ["watch", "wristwatch"],
    "belt": ["belt", "waist belt"],
    "leg warmer": ["leg warmer", "legwear"],
    "tights, stockings": ["tights", "stockings", "hosiery"],
    "sock": ["sock", "socks"],
    "shoe": ["shoe", "footwear", "sneaker"],
    "bag, wallet": ["bag", "handbag", "purse", "wallet"],
    "scarf": ["scarf", "shawl", "wrap"],
    "umbrella": ["umbrella", "parasol"],
    "hood": ["hood", "hoodie part"],
    "collar": ["collar", "neckline part"],
    "lapel": ["lapel", "jacket lapel"],
    "epaulette": ["epaulette", "shoulder detail"],
    "sleeve": ["sleeve", "arm part"],
    "pocket": ["pocket", "clothing pocket"],
    "neckline": ["neckline", "neck opening"],
    "buckle": ["buckle", "fastener"],
    "zipper": ["zipper", "zip fastener"],
    "applique": ["applique", "decoration"],
    "bead": ["bead", "embellishment"],
    "bow": ["bow", "ribbon bow"],
    "flower": ["flower", "floral decoration"],
    "fringe": ["fringe", "tassel trim"],
    "ribbon": ["ribbon", "trim"],
    "rivet": ["rivet", "metal stud"],
    "ruffle": ["ruffle", "frill"],
    "sequin": ["sequin", "sparkle"],
    "tassel": ["tassel", "hanging ornament"],
}


def primary_category_name(name: str) -> str:
    return name.split(",")[0].strip().lower()


def _join_phrases(parts: List[str]) -> str:
    cleaned = [p.strip().lower() for p in parts if p and p.strip()]
    if not cleaned:
        return "object."
    return " . ".join(cleaned) + "."


def build_structured_prompt(category_name: str, strategy: str) -> str:
    """Build a GDINO caption from Fashionpedia category name and strategy id."""
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(PROMPT_STRATEGIES)}")

    primary = primary_category_name(category_name)

    if strategy == "baseline":
        return f"{primary}."

    if strategy == "descriptive":
        return f"fashion {primary} clothing garment."

    if strategy == "synonyms":
        parts = [p.strip().lower() for p in category_name.split(",")]
        return _join_phrases(parts)

    if strategy == "multi_term":
        terms = CATEGORY_MULTI_TERMS.get(category_name)
        if terms is None:
            terms = [primary, "apparel", "garment", "clothing"]
        return _join_phrases(terms)

    if strategy == "ecommerce":
        return f"product . {primary} . fashion apparel . clothing item ."

    raise ValueError(strategy)


def all_strategies() -> List[str]:
    return list(PROMPT_STRATEGIES.keys())
