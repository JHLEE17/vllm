from huggingface_hub import snapshot_download

# sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
sql_lora_path = "/home/jovyan/vol-1/models/llama-2-7b-sql-lora-test"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# llm = LLM(model="/home/jovyan/vol-1/models/Llama-2-7b-hf", enable_lora=True, max_num_seqs=16)#, dtype='bfloat16')
llm = LLM(model="/home/jovyan/vol-1/models/Meta-Llama-3-8B-Instruct", enable_lora=True, max_num_seqs=16)#, dtype='float16')


sampling_params = SamplingParams(
    temperature=0,
    max_tokens=128,
    # stop=["[/assistant]"]
)

# prompts = [
#      "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
#      "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
# ]
# prompts = [
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_95 (one_mora VARCHAR, gloss VARCHAR, accented_mora VARCHAR)\n\n question: What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant]",  # noqa: E501
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_60 (pick INTEGER, former_wnba_team VARCHAR)\n\n question: What pick was a player that previously played for the Minnesota Lynx? [/user] [assistant]",  # noqa: E501
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]"  # noqa: E501
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE employee (id INTEGER, name VARCHAR, department_id INTEGER);\n\n question: Which employees belong to the department with id 3? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE sales (product_id INTEGER, amount_sold INTEGER);\n\n question: What is the total amount sold for product_id 101? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE student (student_id VARCHAR, name VARCHAR, major VARCHAR);\n\n question: List the names of students majoring in Computer Science. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE orders (order_id INTEGER, order_date DATE, customer_id INTEGER);\n\n question: Find all orders placed in January 2024. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE flights (flight_id VARCHAR, departure VARCHAR, destination VARCHAR);\n\n question: List all flights departing from JFK. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE books (isbn VARCHAR, title VARCHAR, author VARCHAR);\n\n question: Find the title of the book with ISBN '1234567890'. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE movie (movie_id INTEGER, title VARCHAR, director VARCHAR);\n\n question: Which movies were directed by Christopher Nolan? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE department (dept_id INTEGER, dept_name VARCHAR);\n\n question: What is the name of the department with dept_id 4? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE project (project_id INTEGER, project_name VARCHAR, budget INTEGER);\n\n question: Which project has the highest budget? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE exam (exam_id INTEGER, subject VARCHAR, score INTEGER);\n\n question: What is the average score for Math exams? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE artist (artist_id INTEGER, name VARCHAR, genre VARCHAR);\n\n question: List the names of artists who perform Jazz. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE team (team_id INTEGER, team_name VARCHAR, city VARCHAR);\n\n question: Which teams are based in New York? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE member (member_id INTEGER, full_name VARCHAR, join_date DATE);\n\n question: Find members who joined after 2020-01-01. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE country (country_id INTEGER, country_name VARCHAR, population INTEGER);\n\n question: Which country has a population greater than 100 million? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE recipe (recipe_id INTEGER, recipe_name VARCHAR, difficulty VARCHAR);\n\n question: List the recipes that are of 'Easy' difficulty. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE vehicle (vehicle_id INTEGER, model VARCHAR, manufacturer VARCHAR);\n\n question: What models are manufactured by Toyota? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE match (match_id INTEGER, team1 VARCHAR, team2 VARCHAR, date DATE);\n\n question: Find matches that were played in 2023. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE contract (contract_id INTEGER, client_id INTEGER, start_date DATE, end_date DATE);\n\n question: Which contracts ended in 2022? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE city (city_id INTEGER, city_name VARCHAR, country_id INTEGER);\n\n question: Which cities are in country with country_id 5? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE lecture (lecture_id INTEGER, topic VARCHAR, duration INTEGER);\n\n question: What is the total duration of all lectures on 'Data Science'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE professor (professor_id INTEGER, name VARCHAR, department VARCHAR);\n\n question: List the names of professors in the 'Physics' department. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE musician (musician_id INTEGER, name VARCHAR, instrument VARCHAR);\n\n question: Which musicians play the piano? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE conference (conf_id INTEGER, title VARCHAR, year INTEGER);\n\n question: List the conferences held in 2023. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE neighborhood (neighborhood_id INTEGER, name VARCHAR, city_id INTEGER);\n\n question: Which neighborhoods are in city with city_id 7? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE task (task_id INTEGER, description VARCHAR, due_date DATE);\n\n question: Find tasks that are due before 2024-01-01. [/user] [assistant]",  #
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE medal (medal_id INTEGER, athlete VARCHAR, event VARCHAR);\n\n question: Which athletes won medals in '100m sprint'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE software (software_id INTEGER, name VARCHAR, version VARCHAR);\n\n question: What is the version of the software named 'SQLPro'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE ocean (ocean_id INTEGER, name VARCHAR, area FLOAT);\n\n question: Which ocean has the largest area? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE volcano (volcano_id INTEGER, name VARCHAR, country VARCHAR);\n\n question: List the names of volcanoes located in Japan. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE fashion_show (show_id INTEGER, name VARCHAR, date DATE);\n\n question: Find the fashion shows held in 2021. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE lecture_series (series_id INTEGER, title VARCHAR, total_lectures INTEGER);\n\n question: How many total lectures are in the series titled 'Machine Learning'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE workshop (workshop_id INTEGER, topic VARCHAR, facilitator VARCHAR);\n\n question: Who is the facilitator for the 'Data Analysis' workshop? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE session (session_id INTEGER, session_name VARCHAR, event_id INTEGER);\n\n question: List the names of sessions for event_id 10. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE charity_event (event_id INTEGER, event_name VARCHAR, amount_raised FLOAT);\n\n question: Which charity event raised the most amount? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE festival (festival_id INTEGER, name VARCHAR, location VARCHAR);\n\n question: Where is the 'Jazz Fest' located? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE innovation (innovation_id INTEGER, name VARCHAR, year INTEGER);\n\n question: List the names of innovations from the year 2020. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE tv_show (show_id INTEGER, title VARCHAR, genre VARCHAR);\n\n question: Which TV shows are categorized under 'Drama'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE gym (gym_id INTEGER, name VARCHAR, location VARCHAR);\n\n question: What is the location of the gym named 'FitLife'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE hospital (hospital_id INTEGER, name VARCHAR, capacity INTEGER);\n\n question: Which hospital has a capacity of over 500 beds? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE song (song_id INTEGER, title VARCHAR, artist VARCHAR);\n\n question: Find the titles of songs by 'The Beatles'. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE cuisine (cuisine_id INTEGER, name VARCHAR, origin VARCHAR);\n\n question: Which cuisines originate from Italy? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE smartphone (phone_id INTEGER, model VARCHAR, brand VARCHAR);\n\n question: List all models of smartphones from the brand 'Samsung'. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE contest (contest_id INTEGER, name VARCHAR, prize_amount FLOAT);\n\n question: What is the prize amount for the contest named 'Hackathon'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE vehicle_service (service_id INTEGER, vehicle_id INTEGER, service_date DATE);\n\n question: When was the last service date for vehicle_id 15? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE ceremony (ceremony_id INTEGER, name VARCHAR, year INTEGER);\n\n question: List the names of ceremonies held in 2022. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE story (story_id INTEGER, title VARCHAR, genre VARCHAR);\n\n question: Which stories are categorized under 'Science Fiction'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE event (event_id INTEGER, event_name VARCHAR, event_date DATE);\n\n question: What is the date of the event named 'Tech Summit'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE trainer (trainer_id INTEGER, name VARCHAR, specialty VARCHAR);\n\n question: Which trainers specialize in 'Strength Training'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE documentary (doc_id INTEGER, title VARCHAR, release_year INTEGER);\n\n question: Find documentaries released in the year 2019. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE institute (institute_id INTEGER, name VARCHAR, country VARCHAR);\n\n question: List the names of institutes located in Canada. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE publication (pub_id INTEGER, title VARCHAR, author VARCHAR);\n\n question: Who is the author of the publication titled 'AI in Modern World'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE festival_event (event_id INTEGER, festival_name VARCHAR, date DATE);\n\n question: Find events for the festival named 'Oktoberfest'. [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE fashion_designer (designer_id INTEGER, name VARCHAR, collection VARCHAR);\n\n question: Which designers have collections named 'Spring 2024'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE heritage_site (site_id INTEGER, name VARCHAR, country VARCHAR);\n\n question: Which heritage sites are located in 'India'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE botanical_garden (garden_id INTEGER, name VARCHAR, location VARCHAR);\n\n question: What is the location of the garden named 'Royal Botanic Gardens'? [/user] [assistant]",  # noqa: E501,
#     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE pet (pet_id INTEGER, name VARCHAR, breed VARCHAR);\n\n question: List the names of pets that are of breed 'Labrador'. [/user] [assistant]",  # noqa: E501,
# ]
prompts = [
        "A robot may not injure a human being",
        "To be or not to be,",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_95 (one_mora VARCHAR, gloss VARCHAR, accented_mora VARCHAR)\n\n question: What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_60 (pick INTEGER, former_wnba_team VARCHAR)\n\n question: What pick was a player that previously played for the Minnesota Lynx? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]"  # noqa: E501
]
outputs = llm.generate(
    prompts,
    sampling_params,
    # lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
)
generated_texts = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# import pdb; pdb.set_trace()

for i in range(len(generated_texts)):
    matching = generated_texts[i] == gaudi[i]
    if matching:
        print(i, 'Pass')
    else:
        print(i, 'Fail!!!!!!!!!!!')
# import pdb; pdb.set_trace()