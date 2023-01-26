run_server:
	python3 -m app.back

run_client:
	python3 -m streamlit run app/front/main.py --server.port 30001 --server.fileWatcherType none

run_app: run_server run_client