import pickle
import streamlit as st
import numpy as np

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.book-container {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}
.book-image {
    margin-right: 20px;
}
.book-title {
    font-size: 18px;
}
.separator {
    margin-top: 20px;
    margin-bottom: 20px;
    border: 0;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

st.title('Book Recommender System')

model = pickle.load(open('Model.pkl','rb'))
book_names = pickle.load(open('Book_Names.pkl','rb'))
final_rating = pickle.load(open('Final_Rating.pkl','rb'))
book_pivot = pickle.load(open('Book_Pivot.pkl','rb'))

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []
    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])
    for name in book_name[0]: 
        ids = np.where(final_rating['Title'] == name)[0][0]
        ids_index.append(ids)
    for idx in ids_index:
        url = final_rating.iloc[idx]['URL']
        poster_url.append(url)
    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    return books_list , poster_url       

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    
    # Display the selected book
    st.markdown("<p class='big-font'>Selected Book:</p>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="book-container">
            <img class="book-image" src="{poster_url[0]}" width="150">
            <p class="book-title">{recommended_books[0]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<p class='big-font'>Recommended Books:</p>", unsafe_allow_html=True)
    for i in range(1, 6):  # Start from 1 to skip the selected book
        st.markdown(
            f"""
            <div class="book-container">
                <img class="book-image" src="{poster_url[i]}" width="150">
                <p class="book-title">{recommended_books[i]}</p>
            </div>
            <hr class="separator">
            """,
            unsafe_allow_html=True
        )