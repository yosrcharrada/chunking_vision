"""A default document + query/ground-truth pairs so the platform works out of
the box.  The document deliberately shifts topic several times so that
entropy-driven boundaries have something to find."""

SAMPLE_DOC = """\
The transformer architecture was introduced in 2017 and reshaped natural language processing. \
Its central mechanism, self-attention, lets every token attend to every other token in a sequence. \
This removed the sequential bottleneck of recurrent networks and made large-scale parallel training practical. \
Within a few years, transformers became the default backbone for language, vision, and audio models alike. \
Scaling these models to billions of parameters revealed surprising emergent abilities in reasoning and translation.

Retrieval-augmented generation, often shortened to RAG, attaches an external knowledge store to a language model. \
Instead of relying only on parameters learned during training, the model retrieves relevant passages at inference time. \
This lets a system answer questions about private or recent documents without retraining the underlying network. \
The quality of a RAG system depends heavily on how the source documents are split into retrievable chunks. \
A retriever embeds the query, compares it against stored chunk vectors, and returns the closest matches.

Chunking is the unglamorous but decisive step in any retrieval pipeline. \
If chunks are too large, retrieval returns diluted passages full of irrelevant text and wastes the context window. \
If chunks are too small, a single idea is scattered across many fragments and no chunk is self-contained. \
Fixed-size chunking ignores meaning entirely and frequently cuts sentences in the middle of an argument. \
Practitioners therefore experiment endlessly with chunk size, overlap, and separators to claw back lost accuracy.

Semantic chunking instead places boundaries where the meaning of the text shifts. \
One way to detect such shifts is information-theoretic: uncertainty about what comes next spikes at a boundary. \
Zellig Harris observed this in 1955 by counting how many different sounds could follow a given sequence. \
Modern methods replace hand counts with neural embeddings and measure semantic distance between adjacent spans. \
A boundary is declared wherever that distance rises sharply above its local neighbourhood.

Entropy quantifies that uncertainty precisely and gives the boundary signal a firm mathematical footing. \
Shannon entropy sums the surprise of each outcome weighted by its probability of occurring. \
Tsallis q-entropy generalises this with a single parameter q that tunes how much rare outcomes count. \
At q equal to one the two measures coincide, so Shannon is just the special case of a broader family. \
Lowering q below one amplifies weak, rare boundaries, while raising it suppresses all but the strongest shifts.

The history of climate science stretches back to the nineteenth century. \
Joseph Fourier first described the greenhouse effect in the 1820s while studying how the atmosphere traps heat. \
Svante Arrhenius later calculated that doubling carbon dioxide would warm the planet by several degrees. \
For decades these ideas remained academic curiosities rather than urgent policy concerns. \
Systematic measurements of atmospheric carbon dioxide only began at Mauna Loa in 1958.

A photosynthesising leaf converts sunlight, water, and carbon dioxide into glucose and oxygen. \
Chlorophyll in the chloroplasts absorbs mostly red and blue light and reflects green, which is why leaves look green. \
The light-dependent reactions split water and release oxygen as a by-product of capturing energy. \
The Calvin cycle then fixes carbon dioxide into sugars using the energy captured in the first stage. \
Together these reactions form the foundation of nearly every food chain on the planet.

The espresso machine forces hot water through a compact puck of finely ground coffee under high pressure. \
The result is a concentrated shot crowned with a reddish-brown foam called crema. \
Baristas adjust the grind size, dose, and extraction time to balance sourness against bitterness. \
A shot pulled too quickly tastes thin and sharp, while one pulled too slowly turns harsh and burnt. \
Milk steamed to a glossy microfoam is what gives a flat white or cappuccino its smooth texture.
"""

SAMPLE_QA = [
    {
        "query": "What is retrieval-augmented generation and why does chunking matter for it?",
        "ground_truth": (
            "RAG attaches an external knowledge store to a language model so it retrieves "
            "relevant passages at inference time; its quality depends on how documents are "
            "split into retrievable chunks."
        ),
    },
    {
        "query": "How does Tsallis q-entropy relate to Shannon entropy?",
        "ground_truth": (
            "Tsallis q-entropy generalises Shannon entropy with a parameter q that tunes how "
            "much rare outcomes count, and at q=1 it reduces exactly to Shannon entropy."
        ),
    },
    {
        "query": "Why is fixed-size chunking a poor choice for retrieval?",
        "ground_truth": (
            "Fixed-size chunking ignores meaning and often cuts sentences mid-argument; chunks "
            "that are too large dilute retrieval and ones that are too small scatter an idea."
        ),
    },
    {
        "query": "How do leaves carry out photosynthesis?",
        "ground_truth": (
            "Chlorophyll absorbs light, the light-dependent reactions split water and release "
            "oxygen, and the Calvin cycle fixes carbon dioxide into sugars."
        ),
    },
    {
        "query": "Who laid the early scientific groundwork for climate science?",
        "ground_truth": (
            "Joseph Fourier described the greenhouse effect in the 1820s and Svante Arrhenius "
            "calculated that doubling carbon dioxide would warm the planet by several degrees."
        ),
    },
    {
        "query": "How does an espresso machine make a shot of coffee?",
        "ground_truth": (
            "An espresso machine forces hot water under high pressure through finely ground "
            "coffee, producing a concentrated shot topped with crema; grind, dose and time are tuned."
        ),
    },
]
