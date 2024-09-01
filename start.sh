docker compose up -d

sleep 16

docker exec -it ollama ollama pull gemma2:2b

python data_init.py