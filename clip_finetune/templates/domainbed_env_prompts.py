from .utils import append_proper_article, get_plural

PACS_prompts = {
    "art painting": [
        "an art painting",
        "a painting",
        "a painting by an artist",
        "a masterpiece painting",
        "a museum painting",
        "a landscape painting",
        "a portrait painting",
        "an oil painting",
        "a watercolor painting",
        "a contemporary painting"
    ],
    "cartoon": [
        "a cartoon",
        "a cartoon character",
        "an animated cartoon",
        "a comic strip",
        "a funny cartoon",
        "a classic cartoon",
        "a Disney cartoon",
        "a Looney Tunes cartoon",
        "a cartoon network show",
        "a cartoon illustration"
    ],
    "real photo": [
        "a real photo",
        "a photograph",
        "a photojournalism",
        "a nature photo",
        "a travel photo",
        "a candid photo",
        "a black and white photo",
        "a landscape photo",
        "a portrait photo",
        "a wildlife photo"
    ],
    "pencil sketch": [
        "a pencil sketch",
        "a sketch",
        "a charcoal drawing",
        "a portrait sketch",
        "a figure sketch",
        "a still life sketch",
        "a gesture sketch",
        "a realistic sketch",
        "a cartoon sketch",
        "a sketch by an artist"
    ]
}

DomainNet_prompts = {
    "clip art drawing": [
        "a clip art drawing",
        "a vector illustration",
        "a cartoon clip art",
        "a digital drawing",
        "a flat design illustration",
        "a simple line drawing",
        "a computer graphic",
        "a colorful clip art",
        "a web icon",
        "a stylized clip art"
    ],
    "infographics": [
        "an infographic",
        "a data visualization",
        "a statistical chart",
        "a diagram",
        "a flowchart",
        "a timeline infographic",
        "a comparison infographic",
        "a map infographic",
        "a pie chart",
        "a bar graph"
    ],
    "art painting": [
        "an art painting",
        "a painting",
        "a painting by an artist",
        "a masterpiece painting",
        "a museum painting",
        "a landscape painting",
        "a portrait painting",
        "an oil painting",
        "a watercolor painting",
        "a contemporary painting"
    ],
    "doodle or quickdraw drawing": [
        "a doodle drawing",
        "a quickdraw sketch",
        "a freehand drawing",
        "a scribble",
        "a casual sketch",
        "a cartoon doodle",
        "a fast drawing",
        "a simple sketch",
        "a rough draft",
        "a spontaneous drawing"
    ],
    "real photo": [
        "a real photo",
        "a photograph",
        "a photojournalism",
        "a nature photo",
        "a travel photo",
        "a candid photo",
        "a black and white photo",
        "a landscape photo",
        "a portrait photo",
        "a wildlife photo"
    ],
    "pencil sketch": [
        "a pencil sketch",
        "a sketch",
        "a charcoal drawing",
        "a portrait sketch",
        "a figure sketch",
        "a still life sketch",
        "a gesture sketch",
        "a realistic sketch",
        "a cartoon sketch",
        "a sketch by an artist"
    ]
}

OfficeHome_prompts = {
    "Art": [
        "an art piece",
        "a work of art",
        "a masterpiece",
        "a sketch",
        "a painting",
        "an artistic creation",
        "a drawing",
        "an ornament",
        "a mural",
        "an illustration"
    ],
    "Clipart": [
        "a clipart image",
        "a vector graphic",
        "a digital drawing",
        "an icon",
        "a cartoon character",
        "a symbol",
        "a logo",
        "a design element",
        "a graphic element",
        "a graphic symbol"
    ],
    "Product": [
        "a product image",
        "a product shot",
        "a product photo",
        "an object on a white background",
        "a product on display",
        "a product from different angles",
        "a product in use",
        "a product packaging",
        "a product label",
        "a product with accessories"
    ],
    "Real-World": [
        "a real-world photo",
        "a street photo",
        "a photojournalism",
        "a travel photo",
        "a landscape photo",
        "a portrait photo",
        "a black and white photo",
        "a nature photo",
        "a photo of everyday life",
        "a photo of urban life"
    ]
}
