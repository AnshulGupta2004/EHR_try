* make a .env file and put your GROQ_API_KEY ="" in it
* cd EHR_TRY
* Then pip install -r requirements.txt
* python T5/main.py --config T5\config\ehrsql\eval\ehrsql_mimic3_t5_base_schema__mimic3_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES "cpu"
Note: First person do not need to change anything from 2nd you have to change model from util/groq_api and T5\config\ehrsql\eval\ehrsql_mimic3_t5_base_schema__mimic3_valid.yaml where you have to put name of the model from hugging face
* If any problem kindly contact anshul. ASAP
