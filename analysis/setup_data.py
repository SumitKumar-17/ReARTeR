"""
Download a small subset of PopQA dataset and build a BM25 corpus for ReARTeR analysis.
PopQA is good for multi-hop reasoning analysis.
"""
import json
import os

# Sample multi-hop questions manually (representative of hotpotqa/musique style)
# These are "gold" QA pairs for analysis
SAMPLE_QUESTIONS = [
    {
        "id": "q001",
        "question": "Who was the president of the United States when the Berlin Wall fell?",
        "golden_answers": ["George H. W. Bush", "George Bush"],
        "supporting_facts": [
            "The Berlin Wall fell on November 9, 1989.",
            "George H. W. Bush was the 41st President of the United States from January 20, 1989, to January 20, 1993."
        ]
    },
    {
        "id": "q002", 
        "question": "What country is the city where Albert Einstein was born located in?",
        "golden_answers": ["Germany"],
        "supporting_facts": [
            "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
            "Ulm is a city in the federal state of Baden-Württemberg, Germany."
        ]
    },
    {
        "id": "q003",
        "question": "What ocean does the longest river in Africa empty into?",
        "golden_answers": ["Atlantic Ocean", "the Atlantic Ocean"],
        "supporting_facts": [
            "The Nile is the longest river in Africa.",
            "The Nile flows northward through northeastern Africa and empties into the Mediterranean Sea."
        ]
    },
    {
        "id": "q004",
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "golden_answers": ["Paris"],
        "supporting_facts": [
            "The Eiffel Tower is located in Paris, France.",
            "Paris is the capital and most populous city of France."
        ]
    },
    {
        "id": "q005",
        "question": "Who invented the telephone and what country was he born in?",
        "golden_answers": ["Alexander Graham Bell", "Scotland"],
        "supporting_facts": [
            "Alexander Graham Bell is credited with inventing the first practical telephone.",
            "Alexander Graham Bell was born in Edinburgh, Scotland on March 3, 1847."
        ]
    },
    {
        "id": "q006",
        "question": "In what year did the company that makes the iPhone release their first personal computer?",
        "golden_answers": ["1976", "1977"],
        "supporting_facts": [
            "The iPhone is made by Apple Inc.",
            "Apple Computer Company was founded on April 1, 1976. The Apple I personal computer was released in 1976."
        ]
    },
    {
        "id": "q007",
        "question": "What language is spoken in the country that won the 2018 FIFA World Cup?",
        "golden_answers": ["French"],
        "supporting_facts": [
            "France won the 2018 FIFA World Cup.",
            "The official language of France is French."
        ]
    },
    {
        "id": "q008",
        "question": "What is the currency of the country where Shakespeare was born?",
        "golden_answers": ["Pound sterling", "British pound", "pound"],
        "supporting_facts": [
            "William Shakespeare was born in Stratford-upon-Avon, England.",
            "England is part of the United Kingdom, which uses the pound sterling as its currency."
        ]
    },
    {
        "id": "q009",
        "question": "What mountain range is Mount Everest part of?",
        "golden_answers": ["Himalayas", "the Himalayas", "Himalayan range"],
        "supporting_facts": [
            "Mount Everest is part of the Himalayan mountain range.",
            "The Himalayas span five countries: Nepal, India, Bhutan, China, and Pakistan."
        ]
    },
    {
        "id": "q010",
        "question": "Who wrote the novel that the movie The Godfather is based on?",
        "golden_answers": ["Mario Puzo"],
        "supporting_facts": [
            "The Godfather is a 1972 American crime film directed by Francis Ford Coppola.",
            "The film is based on the 1969 novel of the same name by Mario Puzo."
        ]
    },
    {
        "id": "q011",
        "question": "What river flows through the city where the Louvre museum is located?",
        "golden_answers": ["Seine", "the Seine", "Seine River"],
        "supporting_facts": [
            "The Louvre is located in Paris, France.",
            "The Seine is a major river that flows through Paris."
        ]
    },
    {
        "id": "q012",
        "question": "What is the atomic number of the element named after Marie Curie's home country?",
        "golden_answers": ["84"],
        "supporting_facts": [
            "Marie Curie was born in Warsaw, Poland.",
            "Polonium is a chemical element named after Poland, Marie Curie's homeland. Polonium has atomic number 84."
        ]
    },
    {
        "id": "q013",
        "question": "What sport is associated with Wimbledon, and in what country is it held?",
        "golden_answers": ["Tennis", "England", "United Kingdom"],
        "supporting_facts": [
            "Wimbledon is a famous tennis tournament.",
            "Wimbledon is held in London, England, United Kingdom."
        ]
    },
    {
        "id": "q014",
        "question": "What is the official language of Brazil?",
        "golden_answers": ["Portuguese"],
        "supporting_facts": [
            "Brazil is the largest country in South America.",
            "The official language of Brazil is Portuguese, due to it being colonized by Portugal."
        ]
    },
    {
        "id": "q015",
        "question": "Who was the first person to walk on the moon and what country was the mission from?",
        "golden_answers": ["Neil Armstrong", "United States", "USA"],
        "supporting_facts": [
            "Neil Armstrong became the first person to walk on the moon on July 20, 1969.",
            "The Apollo 11 mission was launched by NASA, the space agency of the United States."
        ]
    }
]

# Build a small knowledge corpus
CORPUS_DOCUMENTS = [
    {"id": "doc001", "contents": "Berlin Wall\nThe Berlin Wall was a guarded concrete barrier that physically and ideologically divided Berlin from 1961 to 1989. The wall fell on November 9, 1989, when the East German government announced that citizens could cross freely."},
    {"id": "doc002", "contents": "George H. W. Bush\nGeorge Herbert Walker Bush was the 41st President of the United States from January 20, 1989, to January 20, 1993. He oversaw the end of the Cold War, German reunification, and the Gulf War."},
    {"id": "doc003", "contents": "Albert Einstein\nAlbert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. He developed the theory of relativity and is one of the most influential physicists of the 20th century."},
    {"id": "doc004", "contents": "Ulm\nUlm is a city in the federal state of Baden-Württemberg, Germany. It is famous as the birthplace of Albert Einstein and has one of the tallest church steeples in the world."},
    {"id": "doc005", "contents": "Nile River\nThe Nile is a major north-flowing river in northeastern Africa, and is the longest river in Africa. The Nile flows northward through northeastern Africa and empties into the Mediterranean Sea at the Nile Delta in Egypt."},
    {"id": "doc006", "contents": "Africa Rivers\nAfrica's major rivers include the Nile (longest), Congo, Niger, and Zambezi. The Nile has historically been the most important river, supporting ancient Egyptian civilization. The Congo empties into the Atlantic Ocean."},
    {"id": "doc007", "contents": "Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."},
    {"id": "doc008", "contents": "Paris\nParis is the capital and most populous city of France. Situated on the Seine River in northern France, it is home to many famous landmarks including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."},
    {"id": "doc009", "contents": "Alexander Graham Bell\nAlexander Graham Bell was a Scottish-American inventor and scientist who is credited with inventing and patenting the first practical telephone in 1876. He was born in Edinburgh, Scotland on March 3, 1847."},
    {"id": "doc010", "contents": "Scotland\nScotland is a country that is part of the United Kingdom. It is located in the northern part of Great Britain. Scotland is known for its castles, whisky, golf, and famous figures like Alexander Graham Bell and Sir Walter Scott."},
    {"id": "doc011", "contents": "Apple Inc.\nApple Inc. is an American multinational technology company that designs, develops, and sells consumer electronics, computer software, and online services. Apple was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company introduced the iPhone in 2007."},
    {"id": "doc012", "contents": "Apple I\nThe Apple I, released in 1976, was the first Apple computer product. It was designed and hand-built by Steve Wozniak. The Apple II, released in 1977, was the first mass-produced personal computer by Apple."},
    {"id": "doc013", "contents": "2018 FIFA World Cup\nThe 2018 FIFA World Cup was an international football tournament held in Russia from June 14 to July 15, 2018. France won the championship by defeating Croatia 4-2 in the final, claiming their second World Cup title."},
    {"id": "doc014", "contents": "French Language\nFrench is a Romance language that evolved from Vulgar Latin. It is the official language of France and is also spoken in many other countries including Belgium, Switzerland, Canada, and numerous African nations."},
    {"id": "doc015", "contents": "William Shakespeare\nWilliam Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language. He was born in Stratford-upon-Avon, England, around April 23, 1564."},
    {"id": "doc016", "contents": "British Currency\nThe pound sterling, often called the British pound or simply the pound, is the official currency of the United Kingdom, including England, Scotland, Wales, and Northern Ireland. The currency symbol is £."},
    {"id": "doc017", "contents": "Mount Everest\nMount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. It lies on the border between Nepal and Tibet (China). It was first summited on May 29, 1953 by Edmund Hillary and Tenzing Norgay."},
    {"id": "doc018", "contents": "Himalayas\nThe Himalayas is a mountain range in Asia separating the plains of the Indian subcontinent from the Tibetan Plateau. The range spans five countries: Nepal, India, Bhutan, China, and Pakistan, and contains many of Earth's highest peaks including Mount Everest."},
    {"id": "doc019", "contents": "The Godfather\nThe Godfather is a 1972 American crime film directed by Francis Ford Coppola, based on the 1969 novel of the same name by Mario Puzo. The film stars Marlon Brando and Al Pacino and is considered one of the greatest films ever made."},
    {"id": "doc020", "contents": "Mario Puzo\nMario Puzo was an American author, screenwriter, and journalist. He is best known for his crime novel The Godfather (1969), which he later co-adapted into the acclaimed 1972 film. He also wrote The Sicilian and many other novels."},
    {"id": "doc021", "contents": "Seine River\nThe Seine is a river in northern France, 775 km long. It rises at Source-Seine in the Côte-d'Or department and flows northwest, passing through Paris, before emptying into the English Channel at Le Havre."},
    {"id": "doc022", "contents": "Louvre Museum\nThe Louvre, or the Louvre Museum, is the world's largest art museum and a historic monument in Paris, France. A central landmark of Paris, it is located on the Right Bank of the Seine in the city's 1st arrondissement."},
    {"id": "doc023", "contents": "Polonium\nPolonium is a chemical element with the symbol Po and atomic number 84. It was discovered by Marie and Pierre Curie. It was the first element discovered by Marie Curie, and it was named after her homeland of Poland."},
    {"id": "doc024", "contents": "Marie Curie\nMarie Skłodowska Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was born in Warsaw, Poland on November 7, 1867. She discovered two elements: polonium and radium."},
    {"id": "doc025", "contents": "Wimbledon Championships\nThe Wimbledon Championships is the oldest tennis tournament in the world and is widely regarded as the most prestigious. It is held at the All England Club in Wimbledon, London, England, United Kingdom, annually since 1877."},
    {"id": "doc026", "contents": "Tennis\nTennis is a racket sport that can be played individually against a single opponent (singles) or between two teams of two players each (doubles). Wimbledon is one of the four Grand Slam tennis tournaments."},
    {"id": "doc027", "contents": "Brazil\nBrazil is the largest country in both South America and Latin America. It is the largest country to have Portuguese as an official language. Brazil was colonized by Portugal from 1500 until its independence in 1822."},
    {"id": "doc028", "contents": "Portuguese Language\nPortuguese is a Western Romance language originating in the Iberian Peninsula. Today, it is the official language of Brazil, Portugal, Angola, Mozambique, Cape Verde, Guinea-Bissau, Equatorial Guinea, and São Tomé and Príncipe."},
    {"id": "doc029", "contents": "Apollo 11\nApollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. Neil Armstrong became the first person to walk on the Moon."},
    {"id": "doc030", "contents": "NASA\nThe National Aeronautics and Space Administration (NASA) is an independent agency of the U.S. federal government responsible for the civil space program and aeronautics research. It was established in 1958 and conducted the Apollo program."},
    # Extra context documents
    {"id": "doc031", "contents": "Mediterranean Sea\nThe Mediterranean Sea is a body of water surrounded by the Mediterranean Basin, connected to the Atlantic Ocean through the Strait of Gibraltar. The Nile River empties into the Mediterranean Sea via the Nile Delta."},
    {"id": "doc032", "contents": "Germany\nGermany is a country in Central Europe. It is a federal republic consisting of 16 states. Ulm, the birthplace of Albert Einstein, is located in the state of Baden-Württemberg in Germany."},
    {"id": "doc033", "contents": "France\nFrance is a country in Western Europe. Its capital is Paris, which is also the largest city. The official language is French. France is known for its art, culture, cuisine, and landmarks like the Eiffel Tower and Louvre."},
    {"id": "doc034", "contents": "iPhone\nThe iPhone is a line of smartphones made and marketed by Apple Inc. The first iPhone was introduced by Steve Jobs on January 9, 2007, and released on June 29, 2007."},
    {"id": "doc035", "contents": "Kingdom of Württemberg\nThe Kingdom of Württemberg was a monarchy in southwestern Germany from 1806 to 1918. Ulm, birthplace of Albert Einstein, was located in Württemberg. Today this region is part of Baden-Württemberg."},
]

# Save dataset
data_dir = os.path.expanduser("~/ReARTeR/analysis/data")
corpus_dir = os.path.expanduser("~/ReARTeR/analysis/corpus")

# Save as dev split (FlashRAG format)
with open(f"{data_dir}/dev.jsonl", "w") as f:
    for item in SAMPLE_QUESTIONS:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(SAMPLE_QUESTIONS)} questions to {data_dir}/dev.jsonl")

# Save corpus
with open(f"{corpus_dir}/mini_corpus.jsonl", "w") as f:
    for doc in CORPUS_DOCUMENTS:
        f.write(json.dumps(doc) + "\n")

print(f"Saved {len(CORPUS_DOCUMENTS)} documents to {corpus_dir}/mini_corpus.jsonl")
print("Dataset and corpus ready!")
