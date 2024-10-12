# 📚 AI Text Detection App  

Welcome to the **AI Text Detection App**! 🚀 This tool helps you classify text intelligently by leveraging NLP techniques and machine learning. Whether you’re detecting spam emails, classifying documents, or identifying suspicious content—this app has your back! 🕵️‍♂️



---

## 🎯 Features  
- **Streamlit-powered** web app for a smooth and interactive UI.  
- **TF-IDF Vectorization** to extract meaningful word features.  
- **Multinomial Naive Bayes** classifier for high-speed text prediction.  
- **Simple Interface**: Enter your text, click, and classify!  

---

## 🛠️ Installation & Setup  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/your-username/ai-text-detection-app.git
cd ai-text-detection-app
```
Step 2: Create a Virtual Environment
It's always a good idea to isolate your project dependencies!

```bash
# On Windows
python -m venv venv
venv\\Scripts\\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate

```
Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
Step 4: Run the App
Fire up the app locally using Streamlit! 🌐

```bash
streamlit run app.py
```

The app will open in your browser at:
http://localhost:8501

📥 Dataset Used
This app is trained on the DAIGT - One Place, All Data dataset available on Kaggle

Dataset Link:
https://www.kaggle.com/datasets/dsluciano/daigt-one-place-all-data


## 🚀 Future Improvements
- **Interactive model training**: Allow users to train the model with their own datasets.
- **Advanced models**: Add support for more sophisticated classifiers like BERT.
- **API integration**: Expose the model as an API endpoint for broader use.

## 💡 Usage Example
- Enter a text snippet in the input box.
- Click "Predict" to see the magic in action! 🪄
The app will display the predicted class (e.g., AI/Human).

## 💻 Technologies Used
- Python 🐍
- Streamlit for UI
- Scikit-learn for machine learning
- NLTK for text processing
- TF-IDF Vectorizer for feature extraction

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue to discuss what you would like to change.

## 🔥 Try it Out!
Have a specific use case? Fork the project, tweak the code, and make it your own AI-powered detection app! 🛠️

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Need Help?
Open an issue on GitHub if you find a bug 🐛 or have a question ❓
Contact me at karan.sardar2111@gmail.com
Happy Coding! 🎉🚀 """
