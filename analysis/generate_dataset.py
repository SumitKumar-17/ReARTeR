"""
Generate synthetic multi-hop QA dataset at scale (50/100/300/500 questions)
plus a rich knowledge corpus. No internet needed — all facts are hardcoded.
"""
import os, json, random

os.environ['JAX_PLATFORMS'] = 'cpu'

DATA_DIR = os.path.expanduser('~/ReARTeR/analysis/data')
CORP_DIR = os.path.expanduser('~/ReARTeR/analysis/corpus')

# ── LARGE KNOWLEDGE BASE ──────────────────────────────────────────────────────
# Each entry: (question, [gold_answers], [supporting_doc_ids])
QA_BANK = [
  # Geography & Capitals
  ("What is the capital of the country where the Eiffel Tower is located?", ["Paris"], ["france","eiffel"]),
  ("What ocean borders the country that contains the Amazon River?", ["Atlantic Ocean","Atlantic"], ["brazil","amazon"]),
  ("What mountain range runs through the country whose capital is Madrid?", ["Pyrenees","Sierra Nevada"], ["spain","pyrenees"]),
  ("What river flows through the city that is the capital of Austria?", ["Danube","Danube River"], ["vienna","danube"]),
  ("What is the largest city in the country where the Colosseum is located?", ["Rome"], ["italy","colosseum"]),
  ("What sea lies to the north of the continent where the Sahara Desert is?", ["Mediterranean Sea","Mediterranean"], ["africa","sahara","mediterranean"]),
  ("What country is Mount Kilimanjaro located in?", ["Tanzania"], ["kilimanjaro","tanzania"]),
  ("What is the capital of the country with the longest coastline?", ["Ottawa"], ["canada","coastline"]),
  ("What river forms part of the border between the USA and Mexico?", ["Rio Grande","Rio Bravo"], ["usa_mexico","rio_grande"]),
  ("What is the official language of the most populous country in South America?", ["Portuguese"], ["brazil","portuguese"]),
  ("In what country is the ancient city of Machu Picchu located?", ["Peru"], ["machu_picchu","peru"]),
  ("What is the currency of the country where the Acropolis is located?", ["Euro"], ["greece","euro"]),
  ("What ocean does the Amazon River empty into?", ["Atlantic Ocean","Atlantic"], ["amazon","atlantic"]),
  ("What is the tallest mountain in Africa?", ["Kilimanjaro","Mount Kilimanjaro"], ["kilimanjaro","africa"]),
  ("What country has the most pyramids in the world?", ["Sudan","Egypt"], ["sudan","pyramids"]),

  # Science & Inventors
  ("What country was the inventor of the telephone born in?", ["Scotland"], ["bell","scotland"]),
  ("What element did Marie Curie name after her home country?", ["Polonium"], ["curie","polonium"]),
  ("What university did Albert Einstein work at when he published his theory of relativity?", ["University of Bern","Bern"], ["einstein","bern"]),
  ("What is the atomic number of the element whose symbol is Au?", ["79"], ["gold","periodic"]),
  ("What gas makes up most of Earth's atmosphere?", ["Nitrogen"], ["atmosphere","nitrogen"]),
  ("What planet is known as the Red Planet?", ["Mars"], ["mars","planets"]),
  ("Who invented the World Wide Web?", ["Tim Berners-Lee","Timothy Berners-Lee"], ["www","berners_lee"]),
  ("What company did Steve Jobs co-found?", ["Apple","Apple Inc."], ["apple","jobs"]),
  ("In what year was the first iPhone released?", ["2007"], ["iphone","apple"]),
  ("What is the chemical symbol for water?", ["H2O"], ["water","chemistry"]),
  ("What scientist proposed the heliocentric model of the solar system?", ["Copernicus","Nicolaus Copernicus"], ["copernicus","heliocentric"]),
  ("What is the speed of light in km/s (approximately)?", ["300000","299792"], ["light","physics"]),
  ("What organ in the human body produces insulin?", ["Pancreas"], ["insulin","pancreas"]),
  ("Who developed the theory of evolution by natural selection?", ["Charles Darwin","Darwin"], ["darwin","evolution"]),
  ("What year did the first humans land on the Moon?", ["1969"], ["apollo","moon"]),

  # History & Politics
  ("Who was the first President of the United States?", ["George Washington"], ["washington","usa_presidents"]),
  ("What war ended with the Treaty of Versailles?", ["World War I","World War 1","WWI"], ["ww1","versailles"]),
  ("In what year did World War II end?", ["1945"], ["ww2","1945"]),
  ("What empire built the Colosseum?", ["Roman Empire","Rome"], ["colosseum","roman"]),
  ("Who was the first female Prime Minister of the United Kingdom?", ["Margaret Thatcher","Thatcher"], ["thatcher","uk_pm"]),
  ("What country was Nelson Mandela the president of?", ["South Africa"], ["mandela","south_africa"]),
  ("In what year did the Soviet Union dissolve?", ["1991"], ["soviet_union","1991"]),
  ("What was the name of the ship that sank after hitting an iceberg in 1912?", ["Titanic","RMS Titanic"], ["titanic","1912"]),
  ("Who wrote the Communist Manifesto?", ["Karl Marx","Marx and Engels","Karl Marx and Friedrich Engels"], ["marx","communist"]),
  ("What ancient wonder was located in Alexandria, Egypt?", ["Library of Alexandria","Lighthouse of Alexandria","Great Library"], ["alexandria","ancient_wonder"]),
  ("What year did the Berlin Wall fall?", ["1989"], ["berlin_wall","1989"]),
  ("Who was the leader of Nazi Germany during World War II?", ["Adolf Hitler","Hitler"], ["hitler","ww2"]),
  ("What revolution began in France in 1789?", ["French Revolution"], ["french_revolution","1789"]),
  ("Which US president abolished slavery?", ["Abraham Lincoln","Lincoln"], ["lincoln","slavery"]),
  ("What country did Christopher Columbus sail for when he reached the Americas?", ["Spain"], ["columbus","spain"]),

  # Literature & Arts
  ("Who wrote the play Romeo and Juliet?", ["William Shakespeare","Shakespeare"], ["shakespeare","romeo"]),
  ("What is the name of Harry Potter's school?", ["Hogwarts","Hogwarts School of Witchcraft and Wizardry"], ["hogwarts","harry_potter"]),
  ("Who painted the Mona Lisa?", ["Leonardo da Vinci","Leonardo"], ["mona_lisa","davinci"]),
  ("What is the title of J.R.R. Tolkien's most famous trilogy?", ["The Lord of the Rings"], ["tolkien","lotr"]),
  ("Who wrote the novel 1984?", ["George Orwell","Orwell"], ["1984","orwell"]),
  ("What country did Ludwig van Beethoven come from?", ["Germany"], ["beethoven","germany"]),
  ("Who wrote the novel Pride and Prejudice?", ["Jane Austen","Austen"], ["pride_prejudice","austen"]),
  ("What artist cut off his own ear?", ["Vincent van Gogh","Van Gogh"], ["vangogh","ear"]),
  ("What is the best-selling book of all time?", ["The Bible","Bible"], ["bible","bestseller"]),
  ("Who composed the Four Seasons?", ["Antonio Vivaldi","Vivaldi"], ["vivaldi","four_seasons"]),
  ("What novel features the character Atticus Finch?", ["To Kill a Mockingbird"], ["mockingbird","atticus"]),
  ("Who wrote Don Quixote?", ["Miguel de Cervantes","Cervantes"], ["quixote","cervantes"]),
  ("What Shakespeare play features the line 'To be or not to be'?", ["Hamlet"], ["hamlet","shakespeare"]),
  ("Who painted the Sistine Chapel ceiling?", ["Michelangelo"], ["sistine","michelangelo"]),
  ("What is the name of the fictional detective created by Arthur Conan Doyle?", ["Sherlock Holmes","Holmes"], ["sherlock","doyle"]),

  # Sports
  ("What country won the 2018 FIFA World Cup?", ["France"], ["france","fifa2018"]),
  ("What sport is played at Wimbledon?", ["Tennis"], ["wimbledon","tennis"]),
  ("How many players are on a basketball team on the court at one time?", ["5","five"], ["basketball","nba"]),
  ("What country has won the most Olympic gold medals overall?", ["United States","USA"], ["usa","olympics"]),
  ("What is the national sport of Japan?", ["Sumo","Sumo wrestling"], ["japan","sumo"]),
  ("In what city were the first modern Olympic Games held?", ["Athens"], ["athens","olympics1896"]),
  ("What is the maximum score in a single game of ten-pin bowling?", ["300"], ["bowling","perfect_game"]),
  ("Which tennis player has won the most Grand Slam titles (male)?", ["Novak Djokovic","Djokovic"], ["djokovic","grandslam"]),
  ("What country invented the game of chess?", ["India"], ["chess","india"]),
  ("How long is a marathon in kilometers (approx)?", ["42","42.195","42 km"], ["marathon","distance"]),

  # Technology & Business
  ("What does the acronym HTML stand for?", ["HyperText Markup Language"], ["html","web"]),
  ("In what country was the company Samsung founded?", ["South Korea"], ["samsung","korea"]),
  ("What year was Google founded?", ["1998"], ["google","1998"]),
  ("Who is the founder of Tesla and SpaceX?", ["Elon Musk"], ["musk","tesla","spacex"]),
  ("What does CPU stand for?", ["Central Processing Unit"], ["cpu","computer"]),
  ("What company created the Android operating system?", ["Google"], ["android","google"]),
  ("In what year was Facebook founded?", ["2004"], ["facebook","2004"]),
  ("What is the most widely used programming language?", ["Python","JavaScript"], ["python","programming"]),
  ("What company makes the PlayStation gaming console?", ["Sony"], ["playstation","sony"]),
  ("What does AI stand for?", ["Artificial Intelligence"], ["ai","technology"]),

  # Food & Culture
  ("What country is known as the birthplace of pizza?", ["Italy"], ["pizza","italy"]),
  ("What is the main ingredient in guacamole?", ["Avocado"], ["guacamole","avocado"]),
  ("What country does sushi originate from?", ["Japan"], ["sushi","japan"]),
  ("What is the national dish of Spain?", ["Paella"], ["paella","spain"]),
  ("What country produces the most coffee in the world?", ["Brazil"], ["coffee","brazil"]),
  ("What type of pasta is shaped like small rice grains?", ["Orzo","Risoni"], ["orzo","pasta"]),
  ("What country is Gouda cheese originally from?", ["Netherlands","Holland"], ["gouda","netherlands"]),
  ("What fruit is used to make wine?", ["Grapes"], ["wine","grapes"]),
  ("What is the national animal of Australia?", ["Red Kangaroo","Kangaroo"], ["australia","kangaroo"]),
  ("What country is famous for inventing chocolate?", ["Switzerland","Belgium","Mexico"], ["chocolate","switzerland"]),

  # Multi-hop chained questions
  ("What is the capital of the country that won the most medals at the 2020 Olympics?", ["Washington D.C.","Washington DC"], ["usa","olympics2020","washington"]),
  ("What language is spoken in the country where the headquarters of Interpol is located?", ["French"], ["interpol","lyon","france"]),
  ("What river flows through the capital of Australia?", ["Molonglo River","Molonglo"], ["canberra","molonglo"]),
  ("Who was the president of the USA when the first iPhone was released?", ["George W. Bush","Bush"], ["iphone2007","bush"]),
  ("What is the currency of the country where the headquarters of the United Nations is located?", ["US Dollar","Dollar"], ["un","newyork","usd"]),
  ("What ocean would you cross to travel from London to New York?", ["Atlantic Ocean","Atlantic"], ["london","newyork","atlantic"]),
  ("What country is the company that makes Nutella headquartered in?", ["Italy"], ["nutella","ferrero","italy"]),
  ("What is the official language of the country that invented the printing press?", ["German"], ["gutenberg","germany","german"]),
  ("Who was the first person to win two Nobel Prizes?", ["Marie Curie"], ["curie","nobel"]),
  ("In what city was the United Nations founded?", ["San Francisco"], ["un","sanfrancisco"]),

  # More geography
  ("What is the longest river in the world?", ["Nile","Nile River"], ["nile","longest_river"]),
  ("What is the largest ocean on Earth?", ["Pacific Ocean","Pacific"], ["pacific","ocean"]),
  ("What country has the most natural lakes?", ["Canada"], ["canada","lakes"]),
  ("What is the deepest lake in the world?", ["Lake Baikal","Baikal"], ["baikal","lake"]),
  ("What country is home to the most volcanoes?", ["Indonesia"], ["indonesia","volcanoes"]),
  ("What is the smallest country in the world?", ["Vatican City","Vatican"], ["vatican","smallest"]),
  ("What is the largest country by area?", ["Russia"], ["russia","largest"]),
  ("What is the most populous country in the world?", ["India","China"], ["india","population"]),
  ("What is the highest waterfall in the world?", ["Angel Falls"], ["angel_falls","venezuela"]),
  ("What desert is the largest hot desert?", ["Sahara","Sahara Desert"], ["sahara","desert"]),
]

# ── KNOWLEDGE CORPUS ──────────────────────────────────────────────────────────
CORPUS = {
  "france": "France\nFrance is a country in Western Europe. Its capital and largest city is Paris. The official language is French. France is home to landmarks like the Eiffel Tower, the Louvre museum, and the Palace of Versailles. France won the 2018 FIFA World Cup.",
  "eiffel": "Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel and built from 1887–1889 as the entrance arch for the 1889 World's Fair. It is 330 meters tall.",
  "brazil": "Brazil\nBrazil is the largest country in South America and in the Southern Hemisphere. Its capital is Brasília and its largest city is São Paulo. The official language is Portuguese. Brazil is the world's largest coffee producer and home to the Amazon River.",
  "amazon": "Amazon River\nThe Amazon River is a major river in South America. It is the largest river by discharge volume and the second-longest river. The Amazon flows through Brazil and empties into the Atlantic Ocean.",
  "spain": "Spain\nSpain is a country in southwestern Europe located on the Iberian Peninsula. Its capital is Madrid. Spain is known for the Pyrenees mountains on its northern border, flamenco music, and paella.",
  "pyrenees": "Pyrenees\nThe Pyrenees is a mountain range straddling the border of Spain and France. The highest peak is Aneto at 3,404 meters. The range forms a natural border between Spain and France.",
  "vienna": "Vienna\nVienna is the capital and largest city of Austria. The Danube River flows through Vienna. Vienna is famous for its imperial palaces, classical music tradition, and coffee house culture.",
  "danube": "Danube River\nThe Danube is the second-longest river in Europe. It flows through 10 countries including Germany, Austria, and Romania, passing through Vienna, the capital of Austria, before emptying into the Black Sea.",
  "italy": "Italy\nItaly is a country in Southern Europe. Its capital is Rome, which is also the largest city. Italy is known for its rich history, including the Roman Empire, Renaissance art, and cuisine. Italian food includes pizza and pasta.",
  "colosseum": "Colosseum\nThe Colosseum is an ancient amphitheater in Rome, Italy. It was built by the Roman Empire between 70-80 AD and could hold 50,000-80,000 spectators. It was used for gladiatorial contests.",
  "roman": "Roman Empire\nThe Roman Empire was one of the largest empires in ancient history, centered on the city of Rome. It built many famous structures including the Colosseum, aqueducts, and roads across Europe.",
  "africa": "Africa\nAfrica is the world's second-largest and second-most populous continent. The Sahara Desert, the world's largest hot desert, is in northern Africa. The Nile is Africa's longest river. Mount Kilimanjaro is Africa's tallest peak.",
  "sahara": "Sahara Desert\nThe Sahara Desert is the world's largest hot desert, located in northern Africa. It spans 9.2 million square kilometers across countries including Egypt, Libya, Algeria, and Morocco.",
  "mediterranean": "Mediterranean Sea\nThe Mediterranean Sea is a body of water connected to the Atlantic Ocean. It borders southern Europe, northern Africa, and western Asia. It lies to the north of the African continent.",
  "kilimanjaro": "Mount Kilimanjaro\nMount Kilimanjaro is a volcanic mountain in Tanzania, East Africa. At 5,895 meters above sea level, it is the highest peak in Africa. It is a free-standing mountain and the world's tallest free-standing mountain.",
  "tanzania": "Tanzania\nTanzania is a country in East Africa. It is home to Mount Kilimanjaro, the highest peak in Africa. The capital is Dodoma and the largest city is Dar es Salaam.",
  "canada": "Canada\nCanada is the second-largest country in the world by total area. Its capital is Ottawa. Canada has the most freshwater lakes of any country in the world, containing about 60% of all freshwater lakes. Canada also has the world's longest coastline.",
  "coastline": "World Coastlines\nCanada has the longest coastline of any country in the world at approximately 202,080 km. It is followed by Norway, Indonesia, Russia, and the Philippines.",
  "usa_mexico": "US-Mexico Border\nThe border between the United States and Mexico stretches about 3,145 km. The Rio Grande (known as Río Bravo in Mexico) forms a significant portion of this border.",
  "rio_grande": "Rio Grande\nThe Rio Grande is a major river that forms part of the border between the United States and Mexico. It flows from Colorado through New Mexico and then forms the Texas-Mexico border before emptying into the Gulf of Mexico.",
  "portuguese": "Portuguese Language\nPortuguese is a Romance language that originated in Portugal. It is the official language of Brazil, Portugal, Angola, Mozambique, and several other countries. Brazil is the most populous Portuguese-speaking country.",
  "machu_picchu": "Machu Picchu\nMachu Picchu is an Incan citadel located in the Andes Mountains in Peru. Built in the 15th century, it is one of the best-preserved examples of Inca civilization and a UNESCO World Heritage Site.",
  "peru": "Peru\nPeru is a country in South America. Its capital is Lima. Peru is home to Machu Picchu, a famous ancient Inca citadel. The Amazon River originates in Peru.",
  "greece": "Greece\nGreece is a country in southeastern Europe. Its capital is Athens. Greece is home to the Acropolis and the Parthenon. Greece uses the Euro as its currency. Ancient Greece is known as the birthplace of democracy.",
  "euro": "Euro\nThe Euro is the official currency of the eurozone, which includes 20 of the 27 European Union member states including Germany, France, Italy, Spain, Greece, and many others.",
  "atlantic": "Atlantic Ocean\nThe Atlantic Ocean is the second-largest ocean in the world. It separates the Americas from Europe and Africa. The Amazon River empties into the Atlantic Ocean. It is bounded to the north by the Arctic Ocean.",
  "bell": "Alexander Graham Bell\nAlexander Graham Bell was a Scottish-American scientist and inventor born in Edinburgh, Scotland in 1847. He is credited with patenting the first practical telephone in 1876.",
  "scotland": "Scotland\nScotland is a country that is part of the United Kingdom. It is located in the northern part of Great Britain. Edinburgh is its capital. Famous Scots include Alexander Graham Bell, James Watt, and Adam Smith.",
  "curie": "Marie Curie\nMarie Curie was a Polish-French physicist born in Warsaw, Poland in 1867. She discovered two elements: polonium (named after her homeland Poland) and radium. She was the first woman to win a Nobel Prize and the first person to win two Nobel Prizes.",
  "polonium": "Polonium\nPolonium is a chemical element with the symbol Po and atomic number 84. It was discovered by Marie Curie and named after her home country Poland. It is a highly radioactive metal.",
  "einstein": "Albert Einstein\nAlbert Einstein was a German-born theoretical physicist born in Ulm, Germany in 1879. He developed the theory of relativity. He worked at the University of Bern in Switzerland when he published his famous papers in 1905.",
  "bern": "University of Bern\nThe University of Bern is a public university in Bern, Switzerland, founded in 1834. Albert Einstein worked as a patent clerk in Bern and later became associated with the university when he published his theories of relativity.",
  "gold": "Gold Element\nGold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79. It is a dense, soft, shiny metal. Au comes from the Latin word 'aurum'.",
  "periodic": "Periodic Table\nThe periodic table is a tabular arrangement of chemical elements. Gold has the symbol Au and atomic number 79. Silver is Ag (argentum) with atomic number 47.",
  "atmosphere": "Earth's Atmosphere\nEarth's atmosphere is composed of approximately 78% nitrogen, 21% oxygen, and 1% other gases including argon and carbon dioxide. Nitrogen is the most abundant gas.",
  "nitrogen": "Nitrogen\nNitrogen is a chemical element with symbol N and atomic number 7. It makes up about 78% of Earth's atmosphere, making it the most abundant gas in the air we breathe.",
  "mars": "Mars\nMars is the fourth planet from the Sun in our solar system. It is known as the Red Planet due to its reddish appearance caused by iron oxide (rust) on its surface. It has two small moons: Phobos and Deimos.",
  "planets": "Solar System Planets\nThe eight planets in our solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Mars is called the Red Planet. Jupiter is the largest planet.",
  "www": "World Wide Web\nThe World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN. He proposed a system of interlinked hypertext documents accessible via the internet.",
  "berners_lee": "Tim Berners-Lee\nSir Timothy John Berners-Lee is a British computer scientist born in 1955. He invented the World Wide Web in 1989 and created the first web browser and web server.",
  "apple": "Apple Inc.\nApple Inc. is an American technology company co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976. It makes the iPhone, iPad, Mac computers, and the Apple Watch.",
  "jobs": "Steve Jobs\nSteve Jobs was an American businessman and inventor who co-founded Apple Inc. in 1976. He led the development of the Macintosh, iPod, iPhone, and iPad.",
  "iphone": "iPhone\nThe iPhone is a line of smartphones designed and marketed by Apple Inc. The first generation iPhone was introduced by Steve Jobs on January 9, 2007 and released on June 29, 2007. At that time, George W. Bush was the US president.",
  "water": "Water Chemistry\nWater is a chemical compound with the formula H₂O. One molecule of water consists of two hydrogen atoms covalently bonded to one oxygen atom. It is the most abundant substance on Earth's surface.",
  "chemistry": "Chemistry Basics\nChemistry is the scientific study of the properties and behavior of matter. Water has the chemical formula H2O. Salt is NaCl (sodium chloride).",
  "copernicus": "Nicolaus Copernicus\nNicolaus Copernicus was a Renaissance-era mathematician and astronomer who formulated a model of the universe that placed the Sun rather than the Earth at its center — the heliocentric model.",
  "heliocentric": "Heliocentric Model\nThe heliocentric model places the Sun at the center of the solar system. It was proposed by Copernicus in 1543 in his work 'De revolutionibus orbium coelestium'. This contradicted the geocentric model.",
  "light": "Speed of Light\nThe speed of light in a vacuum is approximately 299,792 kilometers per second (roughly 300,000 km/s). It is denoted by the letter c and is a fundamental constant of physics.",
  "physics": "Physics Constants\nKey physics constants: speed of light = ~300,000 km/s, gravitational constant G = 6.674×10⁻¹¹ N⋅m²/kg², Planck's constant h = 6.626×10⁻³⁴ J⋅s.",
  "insulin": "Insulin\nInsulin is a peptide hormone produced by the pancreas. It regulates blood glucose levels. Deficiency or ineffective use of insulin leads to diabetes mellitus.",
  "pancreas": "Pancreas\nThe pancreas is an organ in the human body located in the abdomen. It produces insulin and glucagon to regulate blood sugar, and also produces digestive enzymes.",
  "darwin": "Charles Darwin\nCharles Darwin was a British naturalist born in 1809. He developed the theory of evolution by natural selection, published in 'On the Origin of Species' in 1859.",
  "evolution": "Theory of Evolution\nThe theory of evolution by natural selection was proposed by Charles Darwin in 1859. It states that all species descend from common ancestors and that natural selection drives evolutionary change.",
  "apollo": "Apollo 11\nApollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969. Astronaut Neil Armstrong was the first person to walk on the Moon. The mission was from the United States, launched by NASA.",
  "moon": "Moon Landing\nThe first crewed Moon landing occurred on July 20, 1969, during NASA's Apollo 11 mission. Neil Armstrong became the first human to walk on the Moon, followed by Buzz Aldrin.",
  "washington": "George Washington\nGeorge Washington was the first President of the United States, serving from 1789 to 1797. He is known as the 'Father of the Nation.' The US capital Washington D.C. is named after him.",
  "usa_presidents": "US Presidents\nThe first US president was George Washington (1789-1797). The 16th was Abraham Lincoln who abolished slavery. The 43rd was George W. Bush who was president when the iPhone launched in 2007.",
  "ww1": "World War I\nWorld War I (1914-1918) was a global conflict centered in Europe. It ended with the signing of the Treaty of Versailles on June 28, 1919, in the Hall of Mirrors at the Palace of Versailles, France.",
  "versailles": "Treaty of Versailles\nThe Treaty of Versailles was signed on June 28, 1919, ending World War I. It was signed in the Palace of Versailles, France, and imposed heavy penalties on Germany.",
  "ww2": "World War II\nWorld War II lasted from 1939 to 1945. It ended in Europe on May 8, 1945 (V-E Day). Adolf Hitler was the leader of Nazi Germany during the war.",
  "thatcher": "Margaret Thatcher\nMargaret Thatcher was a British stateswoman who served as Prime Minister of the United Kingdom from 1979 to 1990. She was the first female Prime Minister of the United Kingdom.",
  "uk_pm": "UK Prime Ministers\nThe UK Prime Ministers have included Margaret Thatcher (1979-1990), the first woman to hold the position, Tony Blair (1997-2007), and Boris Johnson (2019-2022).",
  "mandela": "Nelson Mandela\nNelson Mandela was a South African anti-apartheid activist who served as the first black President of South Africa from 1994 to 1999. He won the Nobel Peace Prize in 1993.",
  "south_africa": "South Africa\nSouth Africa is a country at the southern tip of Africa. Nelson Mandela was its first black president from 1994-1999. Its capital cities are Pretoria (executive), Cape Town (legislative), and Bloemfontein (judicial).",
  "soviet_union": "Soviet Union\nThe Soviet Union (USSR) was a federal socialist state that existed from 1922 to 1991. It dissolved on December 25, 1991 when Mikhail Gorbachev resigned as president.",
  "titanic": "RMS Titanic\nThe RMS Titanic was a British passenger liner that sank on April 15, 1912, after hitting an iceberg in the North Atlantic Ocean. Of the 2,224 passengers and crew, more than 1,500 died.",
  "marx": "Karl Marx\nKarl Marx was a German philosopher and economist born in 1818. Together with Friedrich Engels, he wrote 'The Communist Manifesto' in 1848. His ideas formed the basis of Marxism.",
  "communist": "Communist Manifesto\nThe Communist Manifesto was written by Karl Marx and Friedrich Engels and published in 1848. It is one of the world's most influential political documents.",
  "alexandria": "Alexandria, Egypt\nAlexandria is a city in northern Egypt on the Mediterranean coast. In ancient times it was home to the Great Library of Alexandria (one of the largest libraries of the ancient world) and the Lighthouse of Alexandria (one of the Seven Wonders).",
  "ancient_wonder": "Seven Wonders of the Ancient World\nThe Seven Wonders of the Ancient World included the Lighthouse of Alexandria in Egypt, the Great Pyramid of Giza, the Hanging Gardens of Babylon, and the Colossus of Rhodes.",
  "berlin_wall": "Berlin Wall\nThe Berlin Wall was a guarded concrete barrier that divided Berlin from 1961 to 1989. It fell on November 9, 1989 when the East German government opened the checkpoints. George H. W. Bush was the US president at the time.",
  "hitler": "Adolf Hitler\nAdolf Hitler was the leader of Nazi Germany from 1933 to 1945. He led Germany into World War II, which ended in 1945 with Germany's defeat.",
  "french_revolution": "French Revolution\nThe French Revolution began in 1789 and lasted until 1799. It transformed France from a monarchy to a republic. Key events included the storming of the Bastille on July 14, 1789.",
  "lincoln": "Abraham Lincoln\nAbraham Lincoln was the 16th President of the United States (1861-1865). He led the country through the Civil War and abolished slavery with the Emancipation Proclamation. He was assassinated in 1865.",
  "slavery": "Abolition of Slavery in USA\nSlavery was abolished in the United States by the 13th Amendment to the Constitution, ratified in 1865. President Abraham Lincoln issued the Emancipation Proclamation in 1863.",
  "columbus": "Christopher Columbus\nChristopher Columbus was an Italian explorer who sailed under the Spanish crown. In 1492, he completed a voyage to the Americas, believing he had reached Asia. He sailed for Queen Isabella I of Spain.",
  "shakespeare": "William Shakespeare\nWilliam Shakespeare was an English playwright and poet born in Stratford-upon-Avon, England in 1564. He wrote famous plays including Hamlet, Romeo and Juliet, and Macbeth.",
  "romeo": "Romeo and Juliet\nRomeo and Juliet is a tragedy written by William Shakespeare around 1594-1596. It tells the story of two young star-crossed lovers from feuding families in Verona, Italy.",
  "hogwarts": "Hogwarts\nHogwarts School of Witchcraft and Wizardry is a fictional boarding school in the Harry Potter series by J.K. Rowling. Harry Potter is sorted into Gryffindor house.",
  "harry_potter": "Harry Potter\nHarry Potter is a series of fantasy novels by British author J.K. Rowling. The main character attends Hogwarts School of Witchcraft and Wizardry.",
  "mona_lisa": "Mona Lisa\nThe Mona Lisa is a famous painting by Leonardo da Vinci, believed to have been painted between 1503 and 1519. It is housed in the Louvre Museum in Paris, France.",
  "davinci": "Leonardo da Vinci\nLeonardo da Vinci was an Italian Renaissance polymath born in 1452. He painted the Mona Lisa and The Last Supper. He was also known as a sculptor, architect, and scientist.",
  "tolkien": "J.R.R. Tolkien\nJohn Ronald Reuel Tolkien was an English author born in 1892. He wrote 'The Hobbit' and 'The Lord of the Rings' trilogy. His fictional world is called Middle-earth.",
  "lotr": "The Lord of the Rings\nThe Lord of the Rings is a high-fantasy novel trilogy written by J.R.R. Tolkien. It was published in three volumes in 1954-1955 and is one of the best-selling novels of all time.",
  "1984": "1984 Novel\n1984 is a dystopian novel by George Orwell, published in 1949. It depicts a totalitarian society under constant surveillance by 'Big Brother'. The main character is Winston Smith.",
  "orwell": "George Orwell\nGeorge Orwell was the pen name of Eric Arthur Blair, an English novelist born in 1903. He wrote 1984 and Animal Farm. His works are known for their opposition to totalitarianism.",
  "beethoven": "Ludwig van Beethoven\nLudwig van Beethoven was a German composer born in Bonn, Germany in 1770. He composed nine symphonies, five piano concertos, and one violin concerto. He continued to compose even after becoming deaf.",
  "germany": "Germany\nGermany is a country in Central Europe. Its capital is Berlin. German is the official language. Germany is known for its contributions to science, philosophy, and music. Birthplace of Beethoven, Goethe, and Albert Einstein.",
  "pride_prejudice": "Pride and Prejudice\nPride and Prejudice is a romantic novel written by Jane Austen, published in 1813. It follows Elizabeth Bennet and her relationship with the wealthy Mr. Darcy.",
  "austen": "Jane Austen\nJane Austen was an English novelist born in 1775. She wrote Pride and Prejudice, Sense and Sensibility, and Emma. Her novels focus on social class and marriage in 19th-century England.",
  "vangogh": "Vincent van Gogh\nVincent van Gogh was a Dutch post-impressionist painter born in 1853. He is known for cutting off part of his own left ear in 1888. Famous works include The Starry Night and Sunflowers.",
  "ear": "Van Gogh's Ear\nIn December 1888, Dutch painter Vincent van Gogh cut off part of his left ear during a mental breakdown in Arles, France. He later painted several self-portraits.",
  "bible": "The Bible\nThe Bible is a collection of sacred texts of Judaism and Christianity. It is the best-selling book of all time, with estimated sales of over 5 billion copies.",
  "bestseller": "Best-Selling Books\nThe Bible is the best-selling book of all time with over 5 billion copies sold. Other top sellers include Don Quixote, A Tale of Two Cities, and The Lord of the Rings.",
  "vivaldi": "Antonio Vivaldi\nAntonio Vivaldi was an Italian Baroque composer born in Venice in 1678. He is best known for 'The Four Seasons', a group of four violin concertos representing the seasons.",
  "four_seasons": "The Four Seasons\nThe Four Seasons is a group of four violin concertos by Antonio Vivaldi, composed in 1723. They are among the most popular and recognizable pieces of Baroque music.",
  "mockingbird": "To Kill a Mockingbird\nTo Kill a Mockingbird is a novel by Harper Lee published in 1960. It features lawyer Atticus Finch defending a Black man accused of rape in the American South during the 1930s.",
  "atticus": "Atticus Finch\nAtticus Finch is the main character in Harper Lee's novel 'To Kill a Mockingbird'. He is a lawyer in Alabama who defends Tom Robinson, a Black man falsely accused of a crime.",
  "quixote": "Don Quixote\nDon Quixote is a Spanish novel written by Miguel de Cervantes, published in two parts in 1605 and 1615. It is widely regarded as the first modern novel and one of the greatest works of literature.",
  "cervantes": "Miguel de Cervantes\nMiguel de Cervantes was a Spanish novelist born in 1547. He wrote Don Quixote, which is considered the first modern novel and one of the greatest works of world literature.",
  "hamlet": "Hamlet\nHamlet is a tragedy written by William Shakespeare. It features the famous soliloquy beginning 'To be or not to be, that is the question.' The play is set in Denmark.",
  "sistine": "Sistine Chapel\nThe Sistine Chapel is a chapel in the Apostolic Palace in Vatican City. Its ceiling was painted by Michelangelo between 1508 and 1512. The ceiling depicts scenes from the Book of Genesis.",
  "michelangelo": "Michelangelo\nMichelangelo di Lodovico Buonarroti Simoni was an Italian sculptor, painter, and architect born in 1475. He painted the Sistine Chapel ceiling and sculpted the David and the Pietà.",
  "sherlock": "Sherlock Holmes\nSherlock Holmes is a fictional detective created by British author Arthur Conan Doyle. He first appeared in 1887. Holmes lives at 221B Baker Street, London, and is known for his logical reasoning.",
  "doyle": "Arthur Conan Doyle\nSir Arthur Conan Doyle was a British author born in 1859. He created the fictional detective Sherlock Holmes, who appeared in four novels and 56 short stories.",
  "fifa2018": "2018 FIFA World Cup\nThe 2018 FIFA World Cup was held in Russia from June 14 to July 15, 2018. France won the tournament by defeating Croatia 4-2 in the final. It was France's second World Cup title.",
  "wimbledon": "Wimbledon Championships\nThe Wimbledon Championships is a tennis tournament held at the All England Lawn Tennis and Croquet Club in London, England, United Kingdom. It is the oldest Grand Slam tennis tournament.",
  "tennis": "Tennis\nTennis is a racket sport played on a rectangular court. Major tournaments include Wimbledon (England), the US Open, the French Open (Roland Garros), and the Australian Open.",
  "basketball": "Basketball\nBasketball is a team sport invented by James Naismith in 1891. Each team has 5 players on the court at one time. The game is played on a rectangular court with a basket (hoop) at each end.",
  "nba": "NBA Basketball\nThe National Basketball Association (NBA) is the premier professional basketball league in the United States. Each team has 5 players on the court. A game consists of four 12-minute quarters.",
  "usa": "United States of America\nThe United States is a country in North America. Its capital is Washington D.C. The USA has won the most Olympic gold medals of any country. It is the third-largest country by area.",
  "olympics": "Olympic Games\nThe modern Olympic Games were first held in Athens, Greece in 1896. The United States has won the most Olympic gold medals of any country. The Games are held every four years.",
  "olympics1896": "1896 Summer Olympics\nThe 1896 Summer Olympics were the first modern Olympic Games, held in Athens, Greece from April 6-15, 1896. They were organized by Pierre de Coubertin and featured athletes from 14 nations.",
  "athens": "Athens\nAthens is the capital and largest city of Greece. It hosted the first modern Olympic Games in 1896. Athens is known as the birthplace of democracy and home to the Acropolis and Parthenon.",
  "japan": "Japan\nJapan is an island country in East Asia. Its capital is Tokyo. Sushi originated in Japan. Sumo wrestling is the national sport of Japan. Japan is known for electronics and automotive industries.",
  "sumo": "Sumo Wrestling\nSumo is a form of wrestling and is Japan's national sport. Two wrestlers (rikishi) compete in a circular ring (dohyo). The wrestler who forces the other out of the ring or touches the ground wins.",
  "djokovic": "Novak Djokovic\nNovak Djokovic is a Serbian professional tennis player born in 1987. He has won the most Grand Slam titles in men's tennis history with 24 titles as of 2024.",
  "grandslam": "Grand Slam Tennis\nThere are four Grand Slam tennis tournaments: Wimbledon (UK), US Open (USA), French Open (France), and Australian Open (Australia). Novak Djokovic holds the record for most men's singles Grand Slam titles.",
  "chess": "Chess\nChess is a strategy board game played on an 8×8 grid. The game originated in India around the 6th century AD as 'chaturanga' before spreading to Persia and eventually Europe.",
  "india": "India\nIndia is a country in South Asia. It is the most populous country in the world. Chess originated in India. India's capital is New Delhi. Hindi and English are official languages.",
  "marathon": "Marathon\nA marathon is a long-distance running race with an official distance of 42.195 kilometers (26.219 miles). The distance originates from the ancient Greek legend of Pheidippides running from Marathon to Athens.",
  "html": "HTML\nHyperText Markup Language (HTML) is the standard markup language for creating web pages. It was created by Tim Berners-Lee in 1991. HTML stands for HyperText Markup Language.",
  "web": "World Wide Web\nThe World Wide Web was invented by Tim Berners-Lee in 1989. HTML (HyperText Markup Language) is the standard language for creating web pages.",
  "samsung": "Samsung\nSamsung is a South Korean multinational conglomerate headquartered in Seoul, South Korea. It was founded in 1938 by Lee Byung-chul. Samsung Electronics is one of the world's largest producers of smartphones.",
  "korea": "South Korea\nSouth Korea is a country in East Asia. Its capital is Seoul. South Korea is home to major companies including Samsung, Hyundai, and LG. It is known for K-pop music and Korean cuisine.",
  "google": "Google\nGoogle was founded on September 4, 1998, by Larry Page and Sergey Brin while they were PhD students at Stanford University. Google created the Android operating system.",
  "musk": "Elon Musk\nElon Musk is a South African-born American entrepreneur. He is the founder, CEO, and chief engineer of SpaceX, and CEO of Tesla, Inc. He was born in Pretoria, South Africa in 1971.",
  "tesla": "Tesla Inc.\nTesla, Inc. is an American electric vehicle and clean energy company founded in 2003. Elon Musk became CEO in 2008. Tesla produces electric cars, batteries, and solar energy products.",
  "spacex": "SpaceX\nSpaceX (Space Exploration Technologies Corp.) is an American aerospace company founded in 2002 by Elon Musk. It develops reusable launch vehicles and spacecraft.",
  "cpu": "Central Processing Unit\nA Central Processing Unit (CPU) is the primary component of a computer that executes instructions. CPU stands for Central Processing Unit. Modern CPUs contain billions of transistors.",
  "computer": "Computers\nA computer is an electronic device that processes information. The main components include the CPU (Central Processing Unit), RAM (memory), and storage. Modern computers use binary code.",
  "android": "Android Operating System\nAndroid is a mobile operating system developed by Google, based on the Linux kernel. It was first released in 2008. Google acquired the Android company in 2005.",
  "facebook": "Facebook\nFacebook (now Meta Platforms) was founded by Mark Zuckerberg and his Harvard College roommates in 2004. It launched on February 4, 2004. As of 2024, it has over 3 billion monthly users.",
  "python": "Python Programming\nPython is a high-level programming language created by Guido van Rossum and released in 1991. It is consistently ranked as one of the most popular programming languages. JavaScript is another widely-used language.",
  "programming": "Programming Languages\nPopular programming languages include Python, JavaScript, Java, C++, and SQL. Python is widely used in data science and AI. JavaScript is the primary language for web development.",
  "playstation": "PlayStation\nPlayStation is a gaming brand created and owned by Sony Interactive Entertainment. The original PlayStation was released in Japan in 1994. Sony is a Japanese multinational company.",
  "sony": "Sony\nSony Corporation is a Japanese multinational conglomerate headquartered in Tokyo, Japan. It makes the PlayStation gaming console, Sony televisions, cameras, and music equipment.",
  "ai": "Artificial Intelligence\nArtificial Intelligence (AI) is the simulation of human intelligence by machines. AI stands for Artificial Intelligence. Key areas include machine learning, neural networks, and natural language processing.",
  "technology": "Technology\nModern technology includes AI (Artificial Intelligence), ML (Machine Learning), IoT (Internet of Things), and blockchain. AI is transforming industries from healthcare to finance.",
  "pizza": "Pizza\nPizza is a dish of Italian origin consisting of a flat, round base of dough topped with tomato sauce, cheese, and various toppings. It originated in Naples, Italy in the 18th century.",
  "guacamole": "Guacamole\nGuacamole is an avocado-based dip originating from Mexico. The main ingredient is avocado, usually mashed and mixed with lime juice, cilantro, onion, and tomatoes.",
  "avocado": "Avocado\nThe avocado is a fruit native to Mexico and Central America. It is the main ingredient in guacamole. It is rich in healthy fats and nutrients.",
  "sushi": "Sushi\nSushi is a traditional Japanese dish of prepared rice with raw or cooked seafood, vegetables, and sometimes tropical fruits. It originated in Japan and spread worldwide in the 20th century.",
  "paella": "Paella\nPaella is a Spanish rice dish originally from Valencia, Spain. It is considered Spain's national dish. Traditional paella contains rice, saffron, vegetables, and meat or seafood.",
  "coffee": "Coffee Production\nBrazil is the world's largest coffee producer, accounting for about one-third of all coffee produced globally. Vietnam and Colombia are the second and third-largest producers.",
  "orzo": "Orzo Pasta\nOrzo is a type of pasta that resembles large grains of rice. The name means 'barley' in Italian. It is also known as 'risoni'. Orzo is used in soups, salads, and casseroles.",
  "pasta": "Pasta\nPasta is a type of food typically made from durum wheat flour mixed with water or eggs. It originated in Italy. Types include spaghetti, penne, orzo (rice-shaped), and fettuccine.",
  "gouda": "Gouda Cheese\nGouda is a Dutch yellow cheese made from cow's milk, named after the city of Gouda in the Netherlands. It is one of the most popular cheeses in the world.",
  "netherlands": "Netherlands\nThe Netherlands is a country in Northwestern Europe. Its capital is Amsterdam. The Netherlands is famous for Gouda and Edam cheeses, windmills, tulips, and Rembrandt.",
  "wine": "Wine\nWine is an alcoholic beverage made from fermented grapes. The fermentation of different varieties of grapes produces red, white, and rosé wines.",
  "grapes": "Grapes\nGrapes are a fruit of the genus Vitis. They are used to make wine through fermentation. Major wine-producing countries include France, Italy, Spain, and the United States.",
  "australia": "Australia\nAustralia is a country and continent in the Southern Hemisphere. Its capital is Canberra. The national animal is the red kangaroo. English is the de facto national language.",
  "kangaroo": "Kangaroo\nThe kangaroo is a marsupial native to Australia. The red kangaroo is the largest marsupial in the world and is the national animal of Australia. Kangaroos carry their young in pouches.",
  "chocolate": "Chocolate History\nChocolate originates from the cacao plant, native to Mesoamerica (Mexico and Central America). The Aztecs used cacao beans as currency. Switzerland and Belgium are famous for chocolate production.",
  "olympics2020": "2020 Tokyo Olympics\nThe 2020 Summer Olympics were held in Tokyo, Japan in 2021 (delayed due to COVID-19). The United States won the most gold medals (39) followed by China (38).",
  "washington_dc": "Washington D.C.\nWashington, D.C. is the capital of the United States. It is located on the east coast between Maryland and Virginia. It is named after George Washington and contains the White House and Capitol.",
  "interpol": "Interpol\nInterpol (International Criminal Police Organization) has its headquarters in Lyon, France. The official languages of Interpol are Arabic, English, French, and Spanish.",
  "lyon": "Lyon, France\nLyon is the third-largest city in France. It is known as the gastronomic capital of France. Interpol (International Criminal Police Organization) is headquartered in Lyon.",
  "canberra": "Canberra\nCanberra is the capital city of Australia. The Molonglo River flows through Canberra. It was purpose-built as the capital, chosen as a compromise between Sydney and Melbourne.",
  "molonglo": "Molonglo River\nThe Molonglo River is a river in New South Wales and the Australian Capital Territory. It flows through Canberra, the capital of Australia, and is a tributary of the Murrumbidgee River.",
  "iphone2007": "First iPhone Release\nThe first iPhone was released on June 29, 2007. At the time of its release, George W. Bush was the 43rd President of the United States (2001-2009).",
  "bush": "George W. Bush\nGeorge W. Bush was the 43rd President of the United States, serving from January 20, 2001 to January 20, 2009. He was president when the first iPhone was released in 2007.",
  "un": "United Nations\nThe United Nations (UN) is an international organization founded in 1945. Its headquarters is in New York City, USA. The US dollar is the primary currency used in New York.",
  "newyork": "New York City\nNew York City is the most populous city in the United States. The United Nations headquarters is located on the East Side of Manhattan. The currency used is the US Dollar.",
  "usd": "US Dollar\nThe United States dollar is the official currency of the United States. The symbol is $ and the abbreviation is USD. It is also widely used internationally.",
  "london": "London\nLondon is the capital and largest city of England and the United Kingdom. It lies on the River Thames. The Atlantic Ocean lies between London and New York City.",
  "gutenberg": "Johannes Gutenberg\nJohannes Gutenberg was a German blacksmith and inventor who introduced the printing press with movable type to Europe around 1440. This revolutionized the spread of information.",
  "german": "German Language\nGerman is the official language of Germany, Austria, and one of the official languages of Switzerland and Luxembourg. It is a West Germanic language spoken by about 100 million people.",
  "nobel": "Nobel Prize\nThe Nobel Prize is awarded annually in Physics, Chemistry, Medicine, Literature, Peace, and Economics. Marie Curie won two Nobel Prizes: Physics in 1903 and Chemistry in 1911, making her the first person to win two Nobel Prizes.",
  "sanfrancisco": "San Francisco\nSan Francisco is a city in California, USA. The United Nations was founded in San Francisco in 1945. The city is known for the Golden Gate Bridge, cable cars, and Alcatraz Island.",
  "nile": "Nile River\nThe Nile is a major north-flowing river in northeastern Africa. At approximately 6,650 km long, it is generally considered the longest river in the world. It flows through Egypt and empties into the Mediterranean Sea.",
  "longest_river": "World's Longest Rivers\nThe Nile River in Africa (6,650 km) is generally considered the world's longest river, though some sources cite the Amazon at 6,400 km. The Nile flows through 11 countries.",
  "pacific": "Pacific Ocean\nThe Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometers. It extends from the Arctic in the north to the Antarctic in the south.",
  "ocean": "World Oceans\nThe five oceans are: Pacific (largest), Atlantic, Indian, Southern (Antarctic), and Arctic. The Pacific Ocean is the largest, covering about one-third of Earth's surface.",
  "lakes": "Canada Lakes\nCanada has more lakes than any other country in the world, containing about 60% of the world's freshwater lakes. Canada has approximately 2 million lakes.",
  "baikal": "Lake Baikal\nLake Baikal is a freshwater rift lake in Siberia, Russia. It is the deepest lake in the world at 1,642 meters and contains about 20% of the world's unfrozen fresh water.",
  "lake": "World's Deepest Lakes\nLake Baikal in Russia is the deepest lake in the world at 1,642 m. It is also the world's largest freshwater lake by volume. The Caspian Sea, though technically a lake, is the largest by surface area.",
  "indonesia": "Indonesia\nIndonesia is a country in Southeast Asia and Oceania. It is an archipelago of over 17,000 islands. Indonesia has the highest number of active volcanoes of any country in the world.",
  "volcanoes": "Volcanoes\nIndonesia has the most volcanoes of any country, with about 147 volcanoes including 76 active ones. This is due to its position on the 'Ring of Fire', a tectonic plate boundary.",
  "vatican": "Vatican City\nVatican City is an independent city-state enclaved within Rome, Italy. It is the smallest sovereign state in the world by both area and population. It is the headquarters of the Roman Catholic Church.",
  "smallest": "Smallest Countries\nThe smallest country in the world by area is Vatican City (0.44 km²), followed by Monaco (2.02 km²) and San Marino (61 km²).",
  "russia": "Russia\nRussia is the largest country in the world by land area, covering over 17 million square kilometers across 11 time zones. Its capital is Moscow. Lake Baikal, the world's deepest lake, is located in Russia.",
  "largest": "Largest Countries\nThe largest countries by land area are: Russia (17.1M km²), Canada (10M km²), USA (9.8M km²), China (9.6M km²), and Brazil (8.5M km²).",
  "population": "World Population\nIndia surpassed China in 2023 to become the most populous country in the world with over 1.4 billion people. China has approximately 1.4 billion people.",
  "angel_falls": "Angel Falls\nAngel Falls (Salto Ángel) in Venezuela is the world's highest uninterrupted waterfall, with a height of 979 meters. It is named after Jimmy Angel, an aviator who flew over it in 1933.",
  "venezuela": "Venezuela\nVenezuela is a country in northern South America. It is home to Angel Falls (Salto Ángel), the world's highest waterfall at 979 meters.",
  "desert": "World Deserts\nThe Sahara Desert is the world's largest hot desert at 9.2 million km². Antarctica is the world's largest cold desert. The Arabian Desert is the second-largest hot desert.",
  "sudan": "Sudan and Egypt Pyramids\nSudan has more pyramids than Egypt — approximately 255 pyramids compared to Egypt's 138. Sudan's ancient Nubian pyramids are smaller but more numerous.",
  "pyramids": "Pyramids\nThe most famous pyramids are the Great Pyramids of Giza in Egypt. However, Sudan (ancient Nubia) actually has more pyramids than Egypt with approximately 255 pyramids.",
  "1945": "End of WWII\nWorld War II ended in 1945. Germany surrendered on May 8, 1945 (V-E Day), and Japan surrendered on September 2, 1945 (V-J Day) following the atomic bombings of Hiroshima and Nagasaki.",
  "1989": "Year 1989\nThe year 1989 saw several major events: the Berlin Wall fell on November 9; the Tiananmen Square protests in China; and Tim Berners-Lee proposed the World Wide Web.",
  "1991": "Year 1991\nIn 1991, the Soviet Union dissolved on December 25. The Gulf War took place. The World Wide Web became publicly available.",
}

# ── BUILD DATASETS ─────────────────────────────────────────────────────────────
random.seed(42)
all_qa = list(QA_BANK)
random.shuffle(all_qa)

print(f"Total questions available: {len(all_qa)}")

# Save each scale
SCALES = [50, 100, 300, 500]
for scale in SCALES:
    items = []
    pool = all_qa * ((scale // len(all_qa)) + 2)  # repeat if needed
    for i, (q, ans, _) in enumerate(pool[:scale]):
        items.append({"id": f"synth_{i:04d}", "question": q, "golden_answers": ans})
    
    out_dir = os.path.join(DATA_DIR, f'synth_{scale}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'dev.jsonl'), 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
    print(f"Saved synth_{scale}/dev.jsonl  ({scale} questions)")

# Save full corpus
docs = [{"id": k, "contents": v} for k, v in CORPUS.items()]
corp_path = os.path.join(CORP_DIR, 'synth_corpus.jsonl')
with open(corp_path, 'w') as f:
    for d in docs:
        f.write(json.dumps(d) + '\n')
print(f"\nSaved corpus: {corp_path} ({len(docs)} documents)")

# Build BM25 index
print("\nBuilding BM25 index...")
import bm25s, Stemmer
stemmer = Stemmer.Stemmer('english')
texts   = [d['contents'] for d in docs]
tokens  = bm25s.tokenize(texts, stemmer=stemmer)
idx     = bm25s.BM25(corpus=docs)
idx.index(tokens)
idx_path = os.path.join(CORP_DIR, 'synth_bm25_index')
os.makedirs(idx_path, exist_ok=True)
idx.save(idx_path, corpus=docs)
print(f"BM25 index saved: {idx_path}")

# Quick retrieval test
q_test = "What country won the 2018 FIFA World Cup?"
qtok = bm25s.tokenize([q_test], stemmer=stemmer)
res, _ = idx.retrieve(qtok, k=2)
print(f"\nRetrieval test: '{q_test}'")
for r in res[0]:
    print(f"  → {r['contents'][:80]}")

print("\n=== All done! ===")
for scale in SCALES:
    print(f"  synth_{scale}/dev.jsonl ready")
