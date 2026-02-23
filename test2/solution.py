import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def clean_text(text):
    """1. Очистка текста от символов, оставляем только слова."""
    if not isinstance(text, str):
        return ""
    # Удаляем все, кроме букв и пробелов, приводим к нижнему регистру
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

def lemmatize_text(text, lemmatizer):
    """2. Пролемматизация текста."""
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def remove_stopwords(text, stop_words):
    """3. Удаление стоп-слов."""
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

# --- ОСНОВНОЙ ПРОЦЕСС ---

# Загрузка данных
# Предполагаем, что файл comments.csv находится в той же папке
df = pd.read_csv('comments.csv')

# Инициализация инструментов
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print("Начинаем обработку текста...")

# Выполнение шагов 1-3
df['clean_text'] = df['text'].apply(clean_text)
df['lemmatized'] = df['clean_text'].apply(lambda x: lemmatize_text(x, lemmatizer))
df['final_text'] = df['lemmatized'].apply(lambda x: remove_stopwords(x, stop_words))

# 4. Вычисление TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['final_text'])
print(f"TF-IDF вычислен. Размер матрицы: {tfidf_matrix.shape}")

# 5. Функция для получения топ-10 популярных слов
def get_top_n_words(corpus, n=10):
    vec = TfidfVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Визуализация топ-10 слов для всего корпуса
top_words_all = get_top_n_words(df['final_text'])
words, scores = zip(*top_words_all)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(scores), y=list(words), palette='viridis')
plt.title('Топ-10 самых популярных слов (по суммарному TF-IDF)')
plt.xlabel('Суммарный вес TF-IDF')
plt.show()

# 6. Топ-10 слов среди положительных (0) и отрицательных (1) комментариев
positive_comments = df[df['toxic'] == 0]['final_text']
negative_comments = df[df['toxic'] == 1]['final_text']

top_positive = get_top_n_words(positive_comments)
top_negative = get_top_n_words(negative_comments)

# Построение графиков для сравнения
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Положительные
pos_words, pos_scores = zip(*top_positive)
sns.barplot(x=list(pos_scores), y=list(pos_words), ax=axes[0], palette='Greens_r')
axes[0].set_title('Топ-10 слов: Положительные (0)')

# Отрицательные
neg_words, neg_scores = zip(*top_negative)
sns.barplot(x=list(neg_scores), y=list(neg_words), ax=axes[1], palette='Reds_r')
axes[1].set_title('Топ-10 слов: Отрицательные (1)')

plt.tight_layout()
plt.show()

print("Задача выполнена успешно.")