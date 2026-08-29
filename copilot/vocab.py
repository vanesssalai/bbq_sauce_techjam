from __future__ import annotations

import re

COLOR_WORDS = [
    "black", "white", "grey", "gray", "red", "blue", "navy", "green",
    "olive", "brown", "tan", "beige", "khaki", "pink", "purple", "yellow",
    "orange", "maroon", "burgundy", "teal", "gold", "golden", "silver",
    "ivory", "cream", "clear", "turquoise", "rainbow", "rose gold",
    "multicolor", "multicolored", "multicoloured", "multi-color", "multi",
]


MATERIAL_WORDS = [
    "cotton", "leather", "wool", "silk", "polyester", "denim", "linen",
    "suede", "cashmere", "nylon", "spandex", "velvet", "canvas", "rayon",
    "fleece", "chiffon", "metal", "stainless steel", "faux leather",
    "plastic", "synthetic", "wood", "silicone", "crystal", "glass",
    "sterling silver", "gemstone", "acrylic", "stone", "alloy", "brass",
    "aluminum", "aluminium", "polyurethane", "rhinestone", "resin", "rubber",
]

SIZE_LETTER_RE = re.compile(r"(?<!['’])\b(XX?S|S|M|L|XX?L|XXX?L|[2-5]X)\b", re.IGNORECASE)
SIZE_NUMERIC_RE = re.compile(r"\bsize[s]?\s*[:\-]?\s*(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE)

SIZE_WORD_MAP: dict[str, str] = {
    "xx-small": "XXS", "xx small": "XXS", "extra extra small": "XXS",
    "x-small": "XS", "x small": "XS", "extra small": "XS",
    "small": "S",
    "medium": "M",
    "large": "L",
    "x-large": "XL", "x large": "XL", "extra large": "XL",
    "xx-large": "XXL", "xx large": "XXL", "extra extra large": "XXL",
}


_SIZE_WORD_ALTERNATION = "|".join(
    sorted((re.escape(k) for k in SIZE_WORD_MAP), key=len, reverse=True)
)


SIZE_WORD_ANCHORED_RE = re.compile(
    rf"(?:,\s*|\bregular\s+)({_SIZE_WORD_ALTERNATION})\s*\)?\s*$", re.IGNORECASE
)

SIZE_WORD_LOOSE_RE = re.compile(rf"\b({_SIZE_WORD_ALTERNATION})\b", re.IGNORECASE)


DEPARTMENT_NORMALIZE: dict[str, str] = {
    "womens": "women", "women": "women",
    "mens": "men", "men": "men",
    "girls": "girls", "boys": "boys",
    "unisexadult": "unisex", "unisexchild": "unisex", "unisexbaby": "unisex",
    "unisex": "unisex",
    "babygirls": "baby", "babyboys": "baby", "baby": "baby",
}
DEPARTMENT_CATEGORY_WORDS: set[str] = {
    "women", "men", "boys", "girls", "unisex", "baby",
}

DEPARTMENT_SYNONYMS: dict[str, str] = {
    "women": "women", "womens": "women", "woman": "women", "ladies": "women",
    "men": "men", "mens": "men", "man": "men", "guys": "men",
    "boys": "boys", "boy": "boys",
    "girls": "girls", "girl": "girls",
    "unisex": "unisex",
    "baby": "baby", "babies": "baby", "infant": "baby",
    "kids": "unisex", "children": "unisex", "child": "unisex",
}


CATEGORY_WORDS = [
    "dress", "shirt", "t-shirt", "tshirt", "blouse", "jeans", "pants",
    "trousers", "shoes", "sneakers", "boots", "sandals", "heels",
    "jacket", "coat", "sweater", "hoodie", "cardigan", "skirt", "shorts",
    "socks", "jewelry", "necklace", "ring", "earrings", "bracelet",
    "watch", "bag", "purse", "backpack", "belt", "hat", "scarf", "gloves",
    "swimsuit", "bikini", "suit", "blazer", "jumpsuit", "leggings",
]

BRAND_WORDS = [
    '2luv', '32 degrees', '5.11', 'adidas', 'adidas originals', 'aerosoles', 'ahnu', 'aldo',
    'alegria by pg lite', 'alex and ani', 'alexander del rossa', 'allegra k', 'amazon collection',
    'amazon essentials', 'amerimark', 'anne klein', "arc'teryx", 'ariat', 'asics', 'avacostume',
    'avidlove', 'baggallini', 'baleaf', 'bandolino', 'baretraps', 'bcbgeneration', 'bcbgmaxazria',
    'bearpaw', 'betsey johnson', 'billabong', 'bioworld', 'birkenstock', 'blencot', 'bling jewelry',
    'blowfish malibu', 'body candy', 'bogs', 'bonnie jean', 'bravado', "breckelle's", 'brinley co',
    'brisco brands', 'brooks', 'bruno marc', 'btfbm', 'buckle-down', 'bulova', 'burton',
    'buttoned down', 'cafepress', 'california costumes', 'callaway', 'calvin klein',
    'cambridge select', 'camel crown', 'carhartt', "carter's", 'casio', 'chaco', 'champion',
    'cherokee', 'chinese laundry', 'cior', 'clarks', 'coach', 'cole haan', 'columbia', 'comfortview',
    'converse', 'coofandy', 'core 10', 'corkys', 'crazy dog t-shirts', 'crocs', 'cupshe', 'dadawen',
    'daily ritual', 'dailyshoes', 'dakine', 'dansko', 'dc comics', 'dearfoams', 'dickies', 'diesel',
    'disguise', 'disney', 'dkny', 'dockers', 'dokotoo', 'doublju', 'dr. martens',
    "dr. scholl's shoes", 'dream pairs', 'earth origins', 'easy spirit', 'easy street', 'ecco',
    'ecowish', 'eddie bauer', 'ekouaer', 'ethika', 'ever faith', 'ever-pretty', 'fanture', 'fergie',
    'fifth sun', 'fila', 'fitflop', 'floerns', 'florsheim', 'fly london', 'footjoy', 'for g and pl',
    'forum novelties', 'fossil', 'fox racing', 'franco sarto', 'free people', 'fruit of the loom',
    'frye', 'fun costumes', 'fun world', 'gem stone king', 'gildan', 'globalwin', 'gloria vanderbilt',
    'goodthreads', 'gore wear', 'grace karin', 'graphics & more', 'haggar', 'hanes',
    'harley-davidson', 'hello kitty', 'helly-hansen', 'hoka one one', 'hot from hollywood', 'hue',
    'hurley', 'hush puppies', 'ice carats', "in'voland", 'inktastic', 'intimo', 'invicta',
    'isotoner', 'ivanka trump', 'izod', 'jambu', 'jbu by jambu', 'jessica simpson', "joe's usa",
    'jordan', 'journee collection', 'k-swiss', 'kate spade new york', 'kavu', 'keds',
    'kenneth cole', 'kenneth cole new york', 'kenneth cole reaction', 'konov', 'la leela', 'lacoste',
    "lands' end", 'lark & ro', 'lauren by ralph lauren', 'leggings depot', 'levaca', "levi's",
    'life is good', 'lifestride', 'london fog', 'lookbookstore', 'louis garneau', 'loungefly',
    'lucky brand', 'lux accessories', 'made by johnny', 'makemechic', 'mangopop', 'marvel',
    'merrell', 'michael antonio', 'michael kors', 'milumia', 'miss me', 'mizuno', 'mordenmiss',
    'mud pie', 'muk luks', 'naturalizer', 'nature breeze', 'nautica', 'new balance', 'nickelodeon',
    'nike', 'nine west', 'novica', 'nydj', "o'neill", 'oakley', 'olukai', "oshkosh b'gosh", 'ouges',
    'outdoor research', 'oxford diamond co', 'pandora', 'pattyboutik', 'pearl izumi', 'peora',
    'pink queen', 'pleaser', 'polar', 'polo ralph lauren', 'port authority', 'prana',
    'premier standard', 'prettygarden', 'puma', 'quiksilver', 'qupid', 'ralph lauren', 'rampage',
    'ray-ban', 'rebecca minkoff', 'reebok', 'regna x', 'rip curl', 'ripple junction', 'roamans',
    'rocket dog', 'rockport', 'romwe', 'ross-simons', 'rothco', 'roxy', "rubie's", 'rvca', 'ryka',
    'sabrina silver', 'saguaro', 'sakkas', 'salomon', 'sam edelman', 'sampeel', 'sanita', 'sanuk',
    'satinior', 'saucony', 'see kai run', 'seiko', 'shein', 'sidefeel', 'silpada', 'silvershake',
    'silverspeck.com', 'simlu', 'skagen', 'skechers', 'smartwool', 'snoozies', 'soly hux', 'sorel',
    'southpole', 'speedo', 'speedy pros', 'sperry', 'spyder', 'stacy adams', 'star wars',
    'steve madden', 'stride rite', 'stuhrling original', 'styleword', 'swarovski', 'swatch',
    'sweatyrocks', 'swiss legend', 'ted baker', 'teva', "the children's place", 'the drop',
    'the mountain', 'the north face', 'the sak', 'thiswear', 'threadrock', 'timberland', 'timex',
    'tissot', 'tommy hilfiger', 'toms', 'travelon', 'trevco', 'tri-mountain', 'tstars',
    'u.s. polo assn.', 'ugg', 'uideazone', 'under armour', 'vaneli', 'vans', 'vera bradley',
    'verdusa', 'victorinox', 'vince camuto', 'vionic', 'viv collection', 'volcom', 'wantdo',
    'watelves', 'wdirara', 'white mountain', 'wolverine', 'woman within', 'woolrich', 'wrangler',
    'yellow box', 'zaful', 'zeagoo', 'zerouv', 'zkess',
]

DISSATISFACTION_CUES = [
    "don't like", "dont like", "not a fan", "not what i wanted",
    "not what i'm looking for", "not what im looking for",
    "none of these", "none of them", "not good enough", "no good",
    "hate these", "these are ugly", "terrible", "awful", "not impressed",
    "not right", "doesn't work", "wont work", "won't work",
    "disappointed", "disappointing", "not quite right", "not helpful",
    "nothing here", "not seeing anything", "worse",
]

SOFT_TAG_WORDS = [
    "fit", "comfort", "comfortable", "durability", "durable", "style",
    "stylish", "casual", "formal", "breathable", "warm", "lightweight",
    "waterproof", "stretchy", "elegant", "trendy", "classic", "modern",
]

BUYING_CUES = [
    "buy", "purchase", "order", "need", "want to get", "looking to buy",
    "get me", "i'll take", "add to cart", "checkout", "asap", "today",
    "for my", "gift for",
]

BROWSING_CUES = [
    "just browsing", "just looking", "show me", "what do you have",
    "any recommendations", "explore", "ideas for", "not sure yet",
    "what's available", "curious about", "options for",
]

OVERRIDE_CUES = [
    "actually", "instead", "never mind", "nevermind", "forget that",
    "forget it", "change my mind", "on second thought", "scratch that",
    "not that", "different", "rather have",
]

NEGATION_CUES = [
    "not ", "no ", "don't want", "dont want", "except", "other than",
    "besides", "excluding",
]