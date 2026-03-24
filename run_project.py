import os

print("Training...")
os.system("python training/train_lstm.py")

print("Evaluating...")
os.system("python training/evaluate.py")

print("Launching dashboard...")
os.system("streamlit run app/app.py")