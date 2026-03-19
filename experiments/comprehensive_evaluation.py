"""
Comprehensive evaluation experiment for GraphRAG system.
Compares performance across different modality combinations and retrieval strategies.
Supports multiple corpora: AttentionPaper, Tesla, and Google.
"""

import os
import re
import sys
import time
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.config import config
from graphrag.retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever, get_graph_context, ask_llm_with_context
from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger

logger = ExperimentLogger("comprehensive_eval")


# ------------------------------------------------------------------ #
# Helper: strip citation markers like [cite: 17, 78]
# ------------------------------------------------------------------ #
_CITE_RE = re.compile(r'\s*\[cite:\s*[\d,\s]+\]', re.IGNORECASE)


def _strip_cite_markers(text: str) -> str:
    """Remove all ``[cite: ...]`` markers from *text*."""
    return _CITE_RE.sub('', text).strip()


# ------------------------------------------------------------------ #
# Corpus base class + concrete implementations
# ------------------------------------------------------------------ #
class Corpus(ABC):
    """Abstract base for a QA corpus used in evaluation."""

    @property
    @abstractmethod
    def corpus_id(self) -> str:
        """Short, unique identifier for this corpus (e.g. 'attention_paper')."""
        ...

    @property
    @abstractmethod
    def questions(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def references(self) -> List[str]:
        ...

    @property
    def relevant_items(self) -> List[List[str]]:
        """Ground-truth relevant node IDs per question (empty by default)."""
        return [[] for _ in self.questions]

    def __len__(self) -> int:
        return len(self.questions)


class AttentionPaperCorpus(Corpus):
    """15 QA pairs about the 'Attention Is All You Need' paper."""

    @property
    def corpus_id(self) -> str:
        return "attention_paper"

    @property
    def questions(self) -> List[str]:
        return [
            "What are the main characteristics of the Transformer architecture?",
            "How does Multi-Head Attention relate to Scaled Dot-Product Attention?",
            "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?",
            "What is the impact of masking in the decoder's self-attention sub-layer?",
            "What role do Positional Encodings play in the Transformer and why are they necessary given the absence of recurrence?",
            "How do residual connections and layer normalization contribute to training stability in the Transformer?",
            "Who authored the Transformer paper and what institutions were they affiliated with?",
            "How does the Transformer Big model compare to ConvS2S and GNMT ensembles on the WMT 2014 English-to-German task?",
            "What are the dimensions and structure of the Feed-Forward sublayer in the Transformer base model?",
            "What dropout rate and where is dropout applied in the Transformer training procedure?",
            "What vocabulary size and encoding scheme is used for WMT 2014 English-to-German versus English-to-French?",
            "What are the number of attention heads and key/value dimensions in the Transformer base versus big model?",
            "What Adam optimizer hyperparameters were used to train the Transformer?",
            "What results did the Transformer achieve on English constituency parsing, and how does it generalize?",
            "Compare the computational complexity per layer of self-attention layers and recurrent layers.",
        ]

    @property
    def references(self) -> List[str]:
        raw = [
            "The Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and using stacked self-attention and point-wise, fully connected layers. [cite: 17, 78]",
            "Multi-Head Attention connects to Scaled Dot-Product Attention by linearly projecting queries, keys, and values h times, and performing the scaled dot-product attention function in parallel on each projected version. [cite: 126, 127]",
            "The Transformer model achieved a new state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU. [cite: 19]",
            "Masking impacts the decoder by preventing positions from attending to subsequent positions, ensuring that predictions for position i can depend only on the known outputs at positions less than i, preserving the auto-regressive property. [cite: 88, 89]",
            "Since the Transformer contains no recurrence or convolution, it has no inherent notion of token order. Positional encodings are added to input embeddings to inject sequence position information. The paper uses sine and cosine functions of different frequencies — sin(pos/10000^(2i/d_model)) and cos(pos/10000^(2i/d_model)) — allowing the model to attend to relative positions and generalize to unseen sequence lengths.",
            "Each sub-layer in the Transformer — Multi-Head Attention and the Feed-Forward Network — employs a residual connection followed by layer normalization, formulated as LayerNorm(x + Sublayer(x)). Residual connections allow gradients to flow directly through the network, mitigating the vanishing gradient problem. Layer normalization stabilizes activations within each layer, enabling training of deep encoder and decoder stacks.",
            "The paper 'Attention Is All You Need' was authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. The authors were affiliated with Google Brain, Google Research, and the University of Toronto (Aidan N. Gomez).",
            "The Transformer (Big) achieved a BLEU score of 28.4 on WMT 2014 English-to-German, outperforming ConvS2S ensemble (26.36 BLEU) and GNMT+RL ensemble (26.30 BLEU). The Transformer Big was also significantly more training-efficient, using fewer FLOPs than comparable ensemble models.",
            "The position-wise Feed-Forward Network consists of two linear transformations with a ReLU activation: FFN(x) = max(0, xW1 + b1)W2 + b2. In the base model, the inner dimension is d_ff = 2048 while the model dimension is d_model = 512. The same feed-forward network is applied independently to each position.",
            "Dropout is applied to the output of each sub-layer before it is added to the sub-layer input and normalized. Dropout is also applied to the sums of the embeddings and positional encodings in both the encoder and decoder stacks. A dropout rate of P_drop = 0.1 is used for the base model.",
            "For English-to-German, the data was encoded using byte-pair encoding with a shared source-target vocabulary of approximately 37,000 tokens. For English-to-French, a word-piece vocabulary of 32,000 tokens was used. Sentence pairs were batched by approximate sequence length.",
            "The base model uses h = 8 parallel attention heads with d_model = 512, giving d_k = d_v = 64 per head. The big model uses h = 16 heads with d_model = 1024, giving d_k = d_v = 64 per head. The big model has 213M parameters compared to 65M for the base model.",
            "Training used the Adam optimizer with β1 = 0.9, β2 = 0.98, and ε = 10^-9. The learning rate followed a warmup schedule: increasing linearly for warmup_steps = 4000 training steps, then decreasing proportionally to the inverse square root of the step number.",
            "The Transformer was evaluated on English constituency parsing using the Wall Street Journal portion of the Penn Treebank. In the WSJ-only setting with a 4-layer model it outperformed all previously reported models except the Recurrent Neural Network Grammar. In the semi-supervised setting using 17 million sentences it achieved 93.3 F1, outperforming all prior work except the Berkeley Parser ensemble, demonstrating the Transformer generalizes well beyond machine translation.",
            "Self-attention layers have a complexity of O(n^2 * d) per layer, while recurrent layers have a complexity of O(n * d^2), making self-attention faster when sequence length n is smaller than representation dimensionality d. [cite: 163, 187, 188, 189]"
        ]
        return [_strip_cite_markers(r) for r in raw]

    @property
    def relevant_items(self) -> List[List[str]]:
        return [
            ["Transformer", "Encoder-Decoder Structure", "Self-Attention Mechanism", "Multi-Head Attention", "Encoder", "Decoder"],
            ["Multi-Head Attention", "Scaled Dot-Product Attention", "Attention Head", "Queries", "Keys", "Values", "Softmax"],
            ["Transformer", "Transformer (Big)", "Transformer (Base Model)", "Gnmt + Rl Ensemble [38]", "Convs2S Ensemble [9]"],
            ["Encoder", "Decoder", "Masking", "Auto-Regressive Property", "Encoder-Decoder Attention"],
            ["Encoder", "Decoder", "Embeddings", "Encoder-Decoder Structure", "Recurrent Neural Networks"],
            ["Encoder", "Decoder", "Encoder-Decoder Structure", "Multi-Head Attention", "Position-Wise Feed-Forward Networks"],
            ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Google Brain", "University Of Toronto"],
            ["Transformer (Big)", "Transformer (Base Model)", "Gnmt + Rl Ensemble [38]", "Convs2S Ensemble [9]", "Convs2S [9]"],
            ["Position-Wise Feed-Forward Networks", "Linear Transformations", "Relu Activation", "Model Dmodel", "Df F", "Inner-Layer"],
            ["Transformer", "Encoder", "Decoder", "Embeddings"],
            ["Transformer", "Encoder", "Decoder"],
            ["Multi-Head Attention", "Scaled Dot-Product Attention", "Transformer (Base Model)", "Transformer (Big)"],
            ["Transformer", "Encoder", "Decoder"],
            ["Transformer", "Transformer (4 Layers) Wsj Only", "Transformer (4 Layers) Semi-Supervised"],
            ["Self-Attention Mechanism", "Recurrent Neural Networks", "Recurrent Language Models", "Convolutional Neural Networks"],
        ]



class TeslaCorpus(Corpus):
    """15 QA pairs about Tesla, derived from data/raw/Tesla.txt."""

    @property
    def corpus_id(self) -> str:
        return "tesla"

    @property
    def questions(self) -> List[str]:
        return [
            "What are Tesla's main product lines as of November 2024?",
            "What was Tesla's total revenue in 2024?",
            "What is the current status and timeline of Tesla's Full Self-Driving technology?",
            "Who leads Tesla and what is the company's leadership structure?",
            "Who are Tesla's main competitors and partners in the EV market?",
            "How did Tesla's acquisitions of Maxwell Technologies and Hibar Systems relate to its battery technology strategy?",
            "What is the relationship between Tesla Energy, SolarCity, and Tesla's energy product portfolio?",
            "How has Tesla's North American Charging Standard affected its relationship with competitors in the EV industry?",
            "Which Gigafactories does Tesla operate and what does each primarily produce?",
            "Who are Tesla's primary battery cell suppliers and what cell formats does each supply?",
            "What NHTSA investigations has Tesla's Autopilot and Full Self-Driving been subject to?",
            "What major legal disputes has Tesla faced involving its founders and employees?",
            "What is Tesla's Optimus robot and what is its development status?",
            "How has Tesla's annual revenue trended from 2017 to 2024?",
            "What is Tesla's insurance business and in which states does it operate?",
        ]

    @property
    def references(self) -> List[str]:
        return [
            "As of November 2024, Tesla offers six vehicle models: Model S, Model X, Model 3, Model Y, Semi, and Cybertruck. Tesla has also announced plans for a second-generation Roadster, the Cybercab, and the Robovan. Beyond vehicles, Tesla sells energy products including the Powerwall, Megapack, Solar Panels, and Solar Roof.",
            "Tesla reported total revenue of US$97.7 billion in 2024, with an operating income of US$7.1 billion and net income of US$7.1 billion. Total assets stood at US$122.1 billion with total equity of US$72.9 billion.",
            "Tesla's Full Self-Driving (Supervised) is an advanced driver-assistance system classified as SAE Level 2 automation, requiring continuous driver supervision. Since 2013, CEO Elon Musk has repeatedly predicted full autonomy (SAE Level 5) within one to three years, but these goals have not been met. All Tesla vehicles produced after April 2019 include Autopilot.",
            "Tesla is led by CEO Elon Musk, who became chief executive in 2008 and owns approximately 13% of the company. Robyn Denholm serves as chair of the board of directors. The company was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning, and is headquartered in Austin, Texas.",
            "Tesla competes in the battery electric vehicle market, where it held a 17.6% market share in 2024. Key partners include battery suppliers Panasonic, CATL, and LG Energy Solution. Toyota and Daimler were former partners.",
            "Tesla acquired Maxwell Technologies in 2019 for its dry electrode technology and ultracapacitor expertise, and acquired Hibar Systems, a battery manufacturing equipment company. Both acquisitions were part of Tesla's strategy to vertically integrate battery cell production and develop the 4680 battery cell program, reducing dependence on external cell suppliers.",
            "Tesla Energy is a subsidiary of Tesla that develops and sells energy storage and generation products including the Powerwall residential battery, Megapack utility-scale battery, and solar products. Tesla Energy acquired SolarCity in 2016, which added solar panel and Tesla Solar Roof products. Tesla Energy also develops the Solar Inverter and operates the Megafactory for Megapack production.",
            "Tesla developed its proprietary charging connector, later standardized as the North American Charging Standard (NACS). Between May 2023 and February 2024, nearly all major North American EV manufacturers announced plans to adopt NACS, effectively making Tesla's connector the industry standard. Toyota also began using Tesla's Supercharger network and NACS.",
            "Tesla operates Gigafactory Nevada (produces Tesla Semi and battery cells with Panasonic), Gigafactory Texas in Austin (produces Cybertruck and Model Y), Gigafactory Shanghai (produces Model 3 and Model Y for Asia-Pacific), Gigafactory Berlin-Brandenburg (produces Model Y for Europe), and the Tesla Fremont Factory in California (original manufacturing hub). Gigafactory New York produces Tesla Solar Roof.",
            "Tesla sources battery cells from three primary suppliers: Panasonic supplies 2170-type cells produced at Gigafactory Nevada and 2170-type cells from Gigafactory Berlin; LG Energy Solution supplies 2170-type cells for Model Y at Gigafactory Shanghai; CATL supplies prismatic cells used in Model 3. Tesla also produces its own 4680-type cells at Gigafactory Texas.",
            "NHTSA has conducted multiple investigations into Tesla's driver assistance systems. A probe was expanded in June 2022 following a series of crashes involving Autopilot. NHTSA investigated phantom braking complaints and battery defects. US federal prosecutors also investigated Autopilot and Full Self-Driving claims. Tesla was required to submit Autopilot crash data under a NHTSA order.",
            "Martin Eberhard, Tesla's original CEO, sued Elon Musk alleging he was pushed out of the company and defamed. The California Department of Fair Employment and Housing sued Tesla over racial discrimination at its Fremont factory. Australian buyers filed a class action over Autopilot misrepresentation. Tesla has also sued former employees including Guangzhi Cao and Alex Khatilov for alleged theft of trade secrets.",
            "Tesla is developing Optimus, a bipedal humanoid robot, as part of its AI and robotics strategy. Elon Musk has positioned Optimus as potentially Tesla's most valuable product long-term. Tesla develops Optimus using the same AI and computer vision capabilities developed for its Full Self-Driving system. The robot has been demonstrated at Tesla AI Day events.",
            "Tesla's revenue grew substantially from $11.76 billion in 2017 to $24.58 billion in 2019, then accelerating to $97.69 billion in 2023. Net income turned consistently positive from 2020 onward after years of losses — net income was -$862M in 2019 but reached $12.56 billion in 2022. Total 2024 revenue was reported at $97.7 billion with net income of $7.1 billion.",
            "Tesla offers its own vehicle insurance product, Tesla Insurance, through Tesla Insurance Services, Inc. The product uses real-time driving behavior data from the vehicle to determine premiums. Tesla Insurance operates in multiple US states including Texas, California, Colorado, Illinois, Minnesota, and others. This vertical integration of insurance is part of Tesla's broader strategy to capture services revenue.",
        ]

    @property
    def relevant_items(self) -> List[List[str]]:
        return [
            ["Model S", "Model X", "Model 3", "Model Y", "Cybertruck", "Tesla Semi", "Powerwall", "Megapack", "Cybercab"],
            ["Tesla, Inc.", "Tesla"],
            ["Full Self-Driving", "Tesla Autopilot", "Autopilot", "Full Self-Driving Beta", "Fsd Software", "Level 2 Automation"],
            ["Elon Musk", "Robyn Denholm", "Martin Eberhard", "Marc Tarpenning", "Tesla, Inc."],
            ["Nacs", "Panasonic", "Solarcity", "Toyota", "Tesla, Inc.", "North American Charging Standard"],
            ["Grohmann Engineering", "Solarcity", "Tesla, Inc.", "Tesla", "Batteries", "4680", "2170"],
            ["Solarcity", "Tesla, Inc.", "Powerwall", "Megapack", "Tesla Automation"],
            ["Nacs", "North American Charging Standard", "Supercharger", "Toyota", "Tesla, Inc."],
            ["Tesla, Inc.", "Tesla"],
            ["Panasonic", "Solarcity", "2170-Type", "4680-Type", "Prismatic Cells", "Tesla, Inc."],
            ["Autopilot", "Tesla Autopilot", "Full Self-Driving", "Full Self-Driving Beta", "Tesla Model S", "Tesla Model X"],
            ["Martin Eberhard", "Elon Musk", "Alex Khatilov", "Tesla, Inc.", "Tesla"],
            ["Tesla, Inc.", "Tesla"],
            ["Tesla, Inc.", "Tesla"],
            ["Tesla, Inc.", "Tesla Insurance Services, Inc."],
        ]



class GoogleCorpus(Corpus):
    """15 QA pairs about Google, derived from data/raw/Google.txt."""

    @property
    def corpus_id(self) -> str:
        return "google"

    @property
    def questions(self) -> List[str]:
        return [
            "How was Google Search originally developed and what algorithm did it use?",
            "What is Google's strategy for generative artificial intelligence?",
            "What is Google's advertising business model and how significant is it to revenue?",
            "What are Google's key products and services across different categories?",
            "What major antitrust actions has Google faced in the US and EU?",
            "How did Google's acquisitions of YouTube and DoubleClick shape its advertising and video dominance?",
            "What is the relationship between DeepMind Technologies, AlphaGo, and Google's generative AI strategy?",
            "What antitrust fines has Google received from European regulators and on what grounds?",
            "How did Google's '20% Innovation Time Off' policy lead to the creation of Gmail, Google News, and AdSense?",
            "What was Eric Schmidt's role at Google and how did the leadership transition to Sundar Pichai occur?",
            "What was Project Nightingale and why did it draw criticism?",
            "What submarine cable infrastructure does Google operate globally?",
            "How does Google's tax strategy using Ireland, the Netherlands, and Bermuda minimize its tax liability?",
            "What were the causes and outcomes of the 2018 Google Walkout?",
            "What was Google's Motorola Mobility acquisition and what happened to it?",
        ]

    @property
    def references(self) -> List[str]:
        return [
            "Google began in January 1996 as a research project by Larry Page and Sergey Brin at Stanford University. They developed the PageRank algorithm, which determined a website's relevance by the number and importance of pages linking to it, rather than counting search term frequency. The search engine was originally nicknamed BackRub because the system checked backlinks. Google was incorporated on September 4, 1998, funded by an initial $100,000 investment from Andy Bechtolsheim.",
            "Following the success of ChatGPT, Google's senior management issued a code red and directed that all products with more than a billion users must incorporate generative AI within months. In March 2023, Google released Bard (now Gemini), a generative AI chatbot. Google has created the text-to-image model Imagen and the text-to-video model Veo. Google also released NotebookLM for synthesizing documents and developed LearnLM, a family of language models serving as personal AI tutors.",
            "Google generates most of its revenues from advertising, including sales of apps, in-app purchases, digital content products, and YouTube. In 2011, 96% of Google's revenue was derived from advertising programs. The primary advertising methods are AdMob, AdSense, and DoubleClick AdExchange. Google Ads allows advertisers to display advertisements through a cost-per-click scheme, while AdSense allows website owners to display ads and earn money per click.",
            "Google's key products span multiple categories: search (Google Search, News, Shopping), email (Gmail), navigation (Google Maps, Waze, Earth), cloud computing (Google Cloud), web browsing (Chrome), video sharing (YouTube), productivity (Workspace including Docs, Sheets, Slides), operating systems (Android, ChromeOS), hardware (Pixel phones, Nest smart home), AI (Google Assistant, Gemini), and cloud storage (Google Drive).",
            "In August 2024, a US federal judge ruled Google held an illegal monopoly over internet search in violation of Section 2 of the Sherman Antitrust Act. In September 2024, the EU Court of Justice imposed a 2.4 billion euro fine on Google for abusing its dominance in the shopping comparison market. The European Commission also fined Google 4.34 billion euros in 2018 for breaching EU antitrust rules related to Android device constraints, and 1.49 billion euros in 2019 for anti-competitive practices in online advertising.",
            "Google acquired YouTube in 2006 for $1.65 billion, gaining the dominant video platform which became a major advertising channel. Google also acquired DoubleClick, bringing the DoubleClick AdExchange into its advertising stack alongside AdSense and AdMob. These acquisitions were later scrutinized in antitrust proceedings; the European Commission fined Google separately over its advertising practices.",
            "Google acquired DeepMind Technologies, which developed AlphaGo — the AI that defeated world champion Go players. Following ChatGPT's success, Google's senior management issued a code red, directing products with over a billion users to integrate generative AI. Google released Bard (later renamed Gemini), and created Imagen (text-to-image), Veo (text-to-video), NotebookLM, and LearnLM as part of this accelerated AI strategy.",
            "The EU Court of Justice upheld a €2.4 billion fine against Google for abusing dominance in the shopping comparison market. The European Commission issued separate fines for anti-competitive practices. CNIL, the French data regulator, fined Google over cookie consent violations. In August 2024, a US federal judge ruled Google held an illegal monopoly in internet search under Section 2 of the Sherman Antitrust Act.",
            "Google's Innovation Time Off policy allowed engineers to spend 20% of their time on personal projects. This policy directly resulted in the creation of Gmail, Google News, AdSense, and Orkut. Gmail launched in 2004 as an invite-only service offering 1GB of storage. AdSense became a core part of Google's advertising revenue model, allowing website publishers to earn revenue by displaying Google ads.",
            "Eric Schmidt joined Google as CEO in 2001, forming an executive trio with founders Larry Page and Sergey Brin. Schmidt served as CEO until 2011, when Larry Page returned as CEO and Schmidt became Executive Chairman. Sundar Pichai, who had introduced Chrome and ChromeOS, became Google CEO in 2015 when Larry Page became CEO of the newly formed Alphabet Inc., with Pichai also becoming Alphabet CEO in 2019.",
            "Project Nightingale was a partnership between Google and Ascension, a US healthcare system, to collect and analyze the personal health data of millions of patients without explicit patient consent. The project involved Google storing identifiable health records. It drew significant criticism over patient privacy, prompted investigation by the Office for Civil Rights of the United States, and became a major example of Google's data privacy controversies.",
            "Google operates several transoceanic submarine cables including Dunant (connecting the United States to mainland Europe), Equiano (connecting Lisbon to Africa), Grace Hopper (connecting the US to Bilbao, Spain and the UK), and Curie (connecting California to South America). These cables form part of Google's strategy to control its own networking infrastructure rather than rely on shared cables.",
            "Google employs aggressive tax avoidance strategies routing profits through Ireland, the Netherlands, and Bermuda — a structure known as the 'Double Irish with a Dutch Sandwich.' This allowed Google to maintain effective tax rates significantly below statutory rates in its major markets. The strategy resulted in an estimated $60 billion in taxes avoided and drew criticism from EU regulators and resulted in reputational damage.",
            "The 2018 Google Walkout saw approximately 20,000 Google employees walk out globally to protest the company's handling of sexual harassment allegations against executives, including a $90 million exit package paid to Andy Rubin despite credible misconduct allegations. Employees demanded an end to forced arbitration, greater transparency, and an employee representative on the board. Google subsequently ended mandatory arbitration for harassment claims.",
            "Google acquired Motorola Mobility in 2012 for $12.5 billion, primarily to gain Motorola's extensive patent portfolio to defend Android against patent litigation. Google retained the patent portfolio but sold the Motorola handset business to Lenovo in 2014 for approximately $2.91 billion, taking a substantial loss on the device business while retaining the strategic patent assets.",
        ]

    @property
    def relevant_items(self) -> List[List[str]]:
        return [
            ["Google", "Larry Page", "Sergey Brin", "Pagerank", "Backrub Search Engine"],
            ["Alphago", "Bard (Now Gemini)", "Imagen", "Veo", "Notebooklm", "Learnlm", "Google"],
            ["Google Ads", "Adsense", "Admob", "Doubleclick Adexchange", "Google", "Alphabet Inc."],
            ["Google Search", "Gmail", "Youtube", "Android", "Google Cloud", "Google Maps", "Gemini"],
            ["Google", "Alphabet Inc.", "€2.4 Billion Fine", "Justice Department"],
            ["Youtube", "Doubleclick", "Doubleclick Adexchange", "Google Ads", "Adsense", "Google"],
            ["Alphago", "Bard (Now Gemini)", "Imagen", "Veo", "Notebooklm", "Learnlm", "Google"],
            ["€2.4 Billion Fine", "Google", "Alphabet Inc."],
            ["Innovation Time Off", "Gmail", "Adsense", "Google News", "Google"],
            ["Eric Schmidt", "Sundar Pichai", "Larry Page", "Google", "Alphabet Inc."],
            ["Google", "Alphabet Inc."],
            ["Google", "Alphabet Inc."],
            ["Google", "Ireland", "Netherlands", "Bermuda"],
            ["Google", "Alphabet Inc."],
            ["Motorola Mobility", "Google", "Alphabet Inc."],
        ]



class SpaceXCorpus(Corpus):
    """15 QA pairs about SpaceX, derived from data/raw/SpaceX.txt."""

    @property
    def corpus_id(self) -> str:
        return "spacex"

    @property
    def questions(self) -> List[str]:
        return [
            "What is SpaceX's Falcon 9 and what makes it significant in launch history?",
            "What is the Starship vehicle and what is its intended purpose?",
            "How did SpaceX's Dragon spacecraft contribute to International Space Station resupply?",
            "What is Starlink and how does it relate to SpaceX's business model?",
            "Who founded SpaceX and what was the original motivation for starting the company?",
            "What were the first three Falcon 1 launches and why were they significant?",
            "How does SpaceX's first-stage booster recovery and reuse work?",
            "What is the relationship between SpaceX and NASA's Artemis Moon program?",
            "What is the Raptor engine and how does it differ from Merlin?",
            "What is Falcon Heavy and what has it been used for?",
            "How does SpaceX's Starlink compete with traditional satellite internet providers?",
            "What regulatory and legal challenges has SpaceX faced for Starship launches?",
            "What is SpaceX's Crew Dragon and how did it end US dependence on Russia for ISS access?",
            "What is SpaceX's long-term vision for Mars colonization?",
            "How does SpaceX generate revenue across its different business lines?",
        ]

    @property
    def references(self) -> List[str]:
        return [
            "Falcon 9 is a two-stage orbital rocket designed by SpaceX for reliable and safe transport of people and payloads to Earth orbit and beyond. It is the first orbital class rocket capable of reflight — its first stage booster is recovered and reused, dramatically reducing launch costs. Falcon 9 has become the most frequently launched American rocket and achieved over 200 successful launches.",
            "Starship is SpaceX's fully reusable super heavy-lift launch system consisting of the Super Heavy booster and the Starship upper stage. It is designed to carry humans and cargo to the Moon, Mars, and beyond. NASA selected Starship as the Human Landing System for the Artemis program to return astronauts to the Moon. Starship is intended to be the most powerful rocket ever built.",
            "SpaceX Dragon is a reusable spacecraft developed to service the ISS under NASA's Commercial Resupply Services contract. Dragon became the first commercial spacecraft to deliver cargo to the ISS in 2012. The Crew Dragon variant (Dragon 2) became the first commercial spacecraft to carry NASA astronauts to the ISS, beginning with the Demo-2 mission in 2020.",
            "Starlink is SpaceX's satellite internet constellation providing broadband internet coverage globally. It consists of thousands of low Earth orbit satellites. Starlink generates significant recurring revenue for SpaceX, helping fund the development of Starship and other programs. Starlink has been deployed in conflict zones including Ukraine and serves customers in over 60 countries.",
            "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs and enabling the colonization of Mars. Musk was motivated partly by NASA's lack of plans for Mars exploration and by a desire to make humanity a multi-planetary species. He invested approximately $100 million of his own money from the PayPal acquisition into SpaceX's founding.",
            "The first three Falcon 1 launches (2006, 2007, 2008) all failed, nearly bankrupting SpaceX. The fourth Falcon 1 launch in September 2008 became the first privately developed liquid-fueled rocket to reach orbit. This success led directly to NASA awarding SpaceX a $1.6 billion Commercial Resupply Services contract, saving the company from bankruptcy.",
            "After stage separation, the Falcon 9 first stage booster performs a series of engine burns to return to the launch site or land on an autonomous drone ship at sea. Grid fins control descent attitude and landing legs deploy for touchdown. Boosters have been reflown over 10 times. This reusability is the primary driver of SpaceX's cost advantage, reducing launch prices significantly compared to expendable rockets.",
            "NASA selected SpaceX's Starship as the Human Landing System for the Artemis program to return astronauts to the lunar surface for the first time since Apollo 17. The contract is worth approximately $2.9 billion. SpaceX is responsible for developing and operating the lunar Starship variant, which will transport astronauts from lunar orbit to the surface and back.",
            "The Raptor is a full-flow staged combustion engine burning liquid methane and liquid oxygen, developed by SpaceX for Starship and Super Heavy. It is the first full-flow staged combustion engine to fly, achieving the highest chamber pressure of any production rocket engine. The Merlin engine, used on Falcon 9 and Falcon Heavy, burns RP-1 (rocket grade kerosene) and liquid oxygen and uses an open-cycle gas generator design.",
            "Falcon Heavy is a heavy-lift launch vehicle consisting of three Falcon 9 cores strapped together, making it the most powerful operational rocket at its debut in 2018. Its first test flight carried Elon Musk's personal Tesla Roadster as a dummy payload into heliocentric orbit. Falcon Heavy has been used for US Air Force national security payloads and commercial geostationary satellites.",
            "Starlink operates in low Earth orbit at approximately 550km altitude, providing latency of 20–40ms compared to 600ms+ for traditional geostationary satellite internet. This low latency makes Starlink suitable for video calls and gaming unlike prior satellite internet. Starlink competes with ViaSat and HughesNet in rural broadband markets and has expanded to aviation and maritime applications.",
            "SpaceX faced significant regulatory hurdles from the FAA for Starship test flights from Boca Chica, Texas. Environmental groups challenged the FAA's environmental review process. The first integrated flight test in April 2023 was delayed due to regulatory approvals and the vehicle exploded during flight. SpaceX has also been in dispute with local communities and environmental advocates over impacts at Boca Chica.",
            "Crew Dragon (Dragon 2) is SpaceX's crewed spacecraft developed under NASA's Commercial Crew Program. Following the retirement of the Space Shuttle in 2011, the US paid Russia approximately $80 million per seat on Soyuz for ISS access. Crew Dragon's first crewed flight in May 2020 (Demo-2) restored US human launch capability. Crew Dragon regularly rotates NASA and international astronaut crews to the ISS.",
            "SpaceX's stated long-term goal is to make humanity multiplanetary by establishing a self-sustaining city on Mars. The company envisions sending thousands of Starship vehicles to Mars to build a colony of one million people within 50 years. The choice of methane as Starship's propellant is deliberate — methane can be synthesized on Mars using the Sabatier reaction from atmospheric CO2 and water ice, enabling in-situ propellant production for return flights.",
            "SpaceX generates revenue from three primary sources: launch services (commercial satellite launches, NASA contracts, US military national security launches via Falcon 9 and Falcon Heavy), Starlink subscriptions (consumer, aviation, maritime, and government plans), and NASA contracts including Commercial Resupply Services and Commercial Crew. Starlink has grown rapidly and is reported to be the largest revenue contributor, helping cross-subsidize Starship development.",
        ]

    @property
    def relevant_items(self) -> List[List[str]]:
        return [
            ["Falcon 9", "Spacex", "Falcon Rockets"],
            ["Starship", "Spacex", "Dragon Spacecraft"],
            ["Dragon", "Dragon 1", "Dragon 2", "Dragon Spacecraft", "Spacex", "Crew Dragon"],
            ["Starlink", "Spacex", "Starlink Satellite Constellation", "Starlink Mini"],
            ["Spacex", "Elon Musk"],
            ["Falcon 1", "Spacex"],
            ["Falcon 9", "Spacex"],
            ["Starship", "Spacex", "Nasa Crewed Spaceflight Artemis Program"],
            ["Raptor Engines", "Falcon 9", "Falcon Heavy", "Spacex"],
            ["Falcon Heavy", "Spacex", "Ses-10"],
            ["Starlink", "Spacex", "Starlink Satellite Constellation"],
            ["Starship", "Spacex"],
            ["Crew Dragon", "Dragon 2", "Demo-2", "Spacex"],
            ["Spacex", "Elon Musk"],
            ["Starlink", "Falcon 9", "Falcon Heavy", "Spacex"],
        ]



def get_all_corpora() -> List[Corpus]:
    """Return all available evaluation corpora."""
    return [
        AttentionPaperCorpus(),
        TeslaCorpus(),
        GoogleCorpus(),
        SpaceXCorpus(),
    ]


def run_baseline_experiment(
    corpus: Corpus,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run baseline GraphRetriever experiment on a single corpus.

    Args:
        corpus: The evaluation corpus to use.

    Returns:
        Tuple of (metrics_summary, per_question_results)
    """
    logger.get_logger().info("=" * 60)
    logger.get_logger().info(
        f"Running BASELINE EXPERIMENT on corpus '{corpus.corpus_id}': "
        "Standard Graph Retrieval"
    )
    logger.get_logger().info("=" * 60)

    retriever = GraphRetriever()
    evaluator = EvaluationPipeline(f"baseline_{corpus.corpus_id}")

    per_question_results = []
    all_metrics = []
    relevant_items_list = corpus.relevant_items

    for i, (question, reference) in enumerate(
        zip(corpus.questions, corpus.references)
    ):
        logger.get_logger().info(
            f"\nQuestion {i+1}/{len(corpus)}: {question[:50]}..."
        )

        try:
            start = time.perf_counter()
            retrieved_context, sources, retrieved_nodes, _relations = (
                get_graph_context(
                    question,
                    retriever.client,
                    retriever.driver,
                    retriever.database,
                )
            )
            answer = ask_llm_with_context(
                question, retrieved_context, retriever.client
            )
            elapsed = time.perf_counter() - start

            assert retrieved_context != answer, (
                "ERROR: retrieved_context must be different from generated "
                "answer (circular evaluation detected)"
            )

            relevant = (
                relevant_items_list[i] if i < len(relevant_items_list) else []
            )

            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=retrieved_context,
                retrieved_items=retrieved_nodes,
                relevant_items=relevant,
                response_time=elapsed,
            )
            metrics.corpus_id = corpus.corpus_id

            all_metrics.append(metrics)

            per_question_results.append({
                "question": question,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "metrics": metrics.to_dict(),
                "response_time": elapsed,
            })

            logger.get_logger().info(
                f"[+] F1: {metrics.retrieval_f1:.3f}, "
                f"Hallucination: {metrics.hallucination_rate:.3f}, "
                f"Response Time: {elapsed:.4f}s"
            )

        except Exception as e:
            logger.get_logger().error(
                f"[!] Error processing question {i+1}: {str(e)}"
            )

    # Aggregate metrics
    n = len(all_metrics)
    metrics_summary = {
        "experiment": "baseline",
        "corpus_id": corpus.corpus_id,
        "num_questions": len(corpus),
        "avg_f1": sum(m.retrieval_f1 for m in all_metrics) / n if n else 0.0,
        "avg_hallucination_rate": (
            sum(m.hallucination_rate for m in all_metrics) / n if n else 0.0
        ),
        "avg_semantic_similarity": (
            sum(m.semantic_similarity for m in all_metrics) / n if n else 0.0
        ),
        "avg_response_time": (
            sum(m.avg_response_time for m in all_metrics) / n if n else 0.0
        ),
    }

    return metrics_summary, per_question_results


def run_multimodal_experiment(
    corpus: Corpus,
    modality_combinations: List[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Run multimodal retrieval ablation study on a single corpus.

    Args:
        corpus: The evaluation corpus to use.
        modality_combinations: List of modality combinations to test.

    Returns:
        Dictionary with results for each combination.
    """
    logger.get_logger().info("\n" + "=" * 60)
    logger.get_logger().info(
        f"Running MULTIMODAL ABLATION STUDY on corpus '{corpus.corpus_id}'"
    )
    logger.get_logger().info("=" * 60)

    retriever = MultimodalGraphRetriever()
    results_by_combo = {}
    relevant_items_list = corpus.relevant_items

    for combo in modality_combinations:
        combo_name = "+".join(combo)
        logger.get_logger().info(
            f"\nTesting modality combination: {combo_name}"
        )

        evaluator = EvaluationPipeline(
            f"multimodal_{corpus.corpus_id}_{combo_name}"
        )
        metrics_list = []

        for i, (question, reference) in enumerate(
            zip(corpus.questions, corpus.references)
        ):
            try:
                start = time.perf_counter()
                answer, metadata = retriever.answer_with_multimodal_context(
                    question, include_modalities=combo
                )
                elapsed = time.perf_counter() - start

                retrieved_context = metadata.get("context", "")
                assert retrieved_context != answer, (
                    "ERROR: retrieved_context must be different from generated "
                    "answer (circular evaluation detected)"
                )

                relevant = (
                    relevant_items_list[i]
                    if i < len(relevant_items_list)
                    else []
                )

                metrics = evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=retrieved_context,
                    retrieved_items=metadata.get("retrieved_nodes", []),
                    relevant_items=relevant,
                    multimodal_context={
                        "text": metadata.get("text_content", ""),
                        "table": metadata.get("table_content", ""),
                        "image": metadata.get("image_content", ""),
                    },
                    response_time=elapsed,
                )
                metrics.corpus_id = corpus.corpus_id

                metrics_list.append(metrics)

            except AssertionError as e:
                logger.get_logger().error(
                    f"[!] Circular evaluation error: {str(e)}"
                )
            except Exception as e:
                logger.get_logger().warning(
                    f"Error processing question: {str(e)}"
                )

        # Aggregate
        if metrics_list:
            n = len(metrics_list)
            results_by_combo[combo_name] = {
                "combination": combo,
                "corpus_id": corpus.corpus_id,
                "num_questions": n,
                "avg_f1": sum(m.retrieval_f1 for m in metrics_list) / n,
                "avg_hallucination": (
                    sum(m.hallucination_rate for m in metrics_list) / n
                ),
                "avg_semantic_sim": (
                    sum(m.semantic_similarity for m in metrics_list) / n
                ),
                "avg_response_time": (
                    sum(m.avg_response_time for m in metrics_list) / n
                ),
                "text_usage": (
                    sum(m.text_modality_usage for m in metrics_list) / n
                ),
                "table_usage": (
                    sum(m.table_modality_usage for m in metrics_list) / n
                ),
                "image_usage": (
                    sum(m.image_modality_usage for m in metrics_list) / n
                ),
            }

            logger.get_logger().info(
                f"  [+] Avg F1: "
                f"{results_by_combo[combo_name]['avg_f1']:.3f}"
            )
            logger.get_logger().info(
                f"  [+] Avg Hallucination: "
                f"{results_by_combo[combo_name]['avg_hallucination']:.3f}"
            )

    return results_by_combo


# ------------------------------------------------------------------ #
# Aggregate helper
# ------------------------------------------------------------------ #
def _aggregate_summaries(
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics across multiple per-corpus summaries."""
    n = len(summaries)
    if n == 0:
        return {}
    total_questions = sum(s.get("num_questions", 0) for s in summaries)
    return {
        "experiment": "baseline_aggregate",
        "num_corpora": n,
        "total_questions": total_questions,
        "avg_f1": sum(s.get("avg_f1", 0) for s in summaries) / n,
        "avg_hallucination_rate": (
            sum(s.get("avg_hallucination_rate", 0) for s in summaries) / n
        ),
        "avg_semantic_similarity": (
            sum(s.get("avg_semantic_similarity", 0) for s in summaries) / n
        ),
        "avg_response_time": (
            sum(s.get("avg_response_time", 0) for s in summaries) / n
        ),
    }


# ------------------------------------------------------------------ #
# Save results
# ------------------------------------------------------------------ #
def save_results(
    per_corpus_baseline: Dict[str, Tuple],
    aggregate_baseline: Dict[str, Any],
    per_corpus_multimodal: Dict[str, Dict],
    output_dir: str = "results",
) -> str:
    """
    Save comprehensive evaluation results.

    Args:
        per_corpus_baseline: {corpus_id: (summary, details)} dicts.
        aggregate_baseline: Aggregate metrics across all corpora.
        per_corpus_multimodal: {corpus_id: multimodal_results} dicts.
        output_dir: Output directory for results.

    Returns:
        Path to results file.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_baseline": aggregate_baseline,
        "per_corpus_baseline": {
            cid: {"summary": data[0], "details": data[1]}
            for cid, data in per_corpus_baseline.items()
        },
        "per_corpus_multimodal": per_corpus_multimodal,
    }

    filepath = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    logger.get_logger().info(f"\n[+] Results saved to {filepath}")
    return filepath


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    """Run comprehensive evaluation experiments across all corpora."""
    logger.get_logger().info("GraphRAG Comprehensive Evaluation")
    logger.get_logger().info(f"Experiment ID: {logger.experiment_name}")

    corpora = get_all_corpora()
    logger.get_logger().info(
        f"Loaded {len(corpora)} corpora: "
        f"{[c.corpus_id for c in corpora]}"
    )

    per_corpus_baseline: Dict[str, Tuple] = {}
    per_corpus_multimodal: Dict[str, Dict] = {}
    baseline_summaries: List[Dict[str, Any]] = []

    modality_combos = [
        ["text"],
        ["text", "table"],
        ["text", "table", "image"],
        ["table"],
        ["image"],
    ]

    for corpus in corpora:
        logger.get_logger().info(
            f"\n{'#' * 60}\n"
            f"# CORPUS: {corpus.corpus_id}  "
            f"({len(corpus)} questions)\n"
            f"{'#' * 60}"
        )

        # Baseline
        baseline_results = run_baseline_experiment(corpus)
        per_corpus_baseline[corpus.corpus_id] = baseline_results
        baseline_summaries.append(baseline_results[0])

        # Multimodal ablation (commented out to save evaluation time)
        # mm_results = run_multimodal_experiment(corpus, modality_combos)
        # per_corpus_multimodal[corpus.corpus_id] = mm_results
        per_corpus_multimodal[corpus.corpus_id] = {}

    # Aggregate baseline metrics
    aggregate_baseline = _aggregate_summaries(baseline_summaries)

    # Save results
    results_file = save_results(
        per_corpus_baseline, aggregate_baseline, per_corpus_multimodal
    )

    # ---- Print summary ---- #
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for cid, (summary, _details) in per_corpus_baseline.items():
        print(f"\n--- Corpus: {cid} ---")
        print(f"  F1 Score:            {summary['avg_f1']:.3f}")
        print(f"  Hallucination Rate:  {summary['avg_hallucination_rate']:.3f}")
        print(f"  Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
        print(f"  Avg Response Time:   {summary['avg_response_time']:.3f}s")

    print(f"\n--- Aggregate (all corpora) ---")
    print(f"  F1 Score:            {aggregate_baseline.get('avg_f1', 0):.3f}")
    print(
        f"  Hallucination Rate:  "
        f"{aggregate_baseline.get('avg_hallucination_rate', 0):.3f}"
    )
    print(
        f"  Semantic Similarity: "
        f"{aggregate_baseline.get('avg_semantic_similarity', 0):.3f}"
    )
    print(
        f"  Avg Response Time:   "
        f"{aggregate_baseline.get('avg_response_time', 0):.3f}s"
    )

    # Best multimodal config across all corpora
    all_mm_entries = []
    for cid, mm_dict in per_corpus_multimodal.items():
        for combo_name, combo_data in mm_dict.items():
            all_mm_entries.append((f"{cid}/{combo_name}", combo_data))

    if all_mm_entries:
        best_config = max(
            all_mm_entries,
            key=lambda x: x[1]['avg_f1'] - x[1]['avg_hallucination'],
        )
        print(f"\nBest Multimodal Configuration:")
        print(f"  Corpus/Modalities: {best_config[0]}")
        print(f"  F1 Score:          {best_config[1]['avg_f1']:.3f}")
        print(
            f"  Hallucination Rate: "
            f"{best_config[1]['avg_hallucination']:.3f}"
        )

    logger.get_logger().info("\n[+] Evaluation complete!")


if __name__ == "__main__":
    main()
