import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt


# PAGE CONFIG

st.set_page_config(page_title="Book Recommendation System", layout="wide")


# CUSTOM CSS

st.markdown("""
<style>
body { background-color: #f8f9fa; }
h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)


# LOAD ARTIFACTS

@st.cache_resource
def load_artifacts():
    with open("user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open("book_map.pkl", "rb") as f:
        book_map = pickle.load(f)
    with open("user_factors.pkl", "rb") as f:
        user_factors = pickle.load(f)
    with open("item_factors.pkl", "rb") as f:
        item_factors = pickle.load(f)
    with open("book_metadata.pkl", "rb") as f:
        book_metadata = pickle.load(f)
    return user_map, book_map, user_factors, item_factors, book_metadata


user_map, book_map, user_factors, item_factors, book_metadata = load_artifacts()
index_to_isbn = {v: k for k, v in book_map.items()}


# RECOMMENDATION LOGIC (BULLETPROOF)

def recommend_books_from_item(isbn, n=5):
    if isbn not in book_map:
        return []

    item_idx = book_map[isbn]
    scores = np.dot(item_factors, item_factors[item_idx])
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices:
        rec_isbn = index_to_isbn.get(idx)
        if rec_isbn and rec_isbn != isbn and rec_isbn in book_metadata:
            results.append(book_metadata[rec_isbn])
        if len(results) == n:
            break

    return results


def recommend_books_for_user(user_id, n=5):
    if user_id not in user_map:
        return [], []

    user_idx = user_map[user_id]
    scores = np.dot(item_factors, user_factors[user_idx])
    sorted_indices = np.argsort(scores)[::-1]

    books, vals = [], []
    for idx in sorted_indices:
        isbn = index_to_isbn.get(idx)
        if isbn and isbn in book_metadata:
            books.append(book_metadata[isbn])
            vals.append(scores[idx])
        if len(books) == n:
            break

    return books, vals


def find_similar_users(user_id, top_n=5):
    if user_id not in user_map:
        return []

    uidx = user_map[user_id]
    sims = np.dot(user_factors, user_factors[uidx])
    idxs = np.argsort(sims)[::-1][1:top_n+1]
    return [list(user_map.keys())[i] for i in idxs]


# SIDEBAR

st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Book-based", "👤 User-based", "📊 User Analytics"]
)


# HOME


if page == "🏠 Home":
    st.title("📚 Book Recommendation System")

    st.markdown("""
    ### 🔍 Project Overview
    This application is a *Book Recommendation System* built using  
    *Collaborative Filtering (Matrix Factorization – SVD)*.

    The goal is to recommend books efficiently based on:
    - User behavior
    - Book similarity
    - Historical ratings

    ---
    ### 🧠 Model Used
    *Truncated SVD (Collaborative Filtering)*  
    - Learns latent user & book representations  
    - Fast inference  
    - Scalable for large datasets  

    ---
    ### 🗂 Data Used
    - *Users*: anonymized user IDs  
    - *Books*: titles, authors, images  
    - *Ratings*: explicit ratings (1–10)  

    ---
    ### 🎯 Key Features
    ✔ Book-based recommendations  
    ✔ User-based recommendations  
    ✔ Cold-start safe logic  
    ✔ User analytics with graphs  
    ✔ Clean UI (no ISBN exposure)  

    ---
    *Built with:* Python, Scikit-learn, Streamlit
    """)


# BOOK-BASED

elif page == "🔍 Book-based":
    st.title("🔍 Book-based Recommendation")

    titles = sorted(set(v["Book-Title"] for v in book_metadata.values()))
    selected = st.selectbox("Select a Book", titles)

    selected_isbn = next(
        (i for i, v in book_metadata.items() if v["Book-Title"] == selected),
        None
    )

    if st.button("📖 Recommend"):
        results = recommend_books_from_item(selected_isbn)

        num_cols = max(1, min(len(results), 5))
        cols = st.columns(num_cols)

        if not results:
            cols[0].info("No strong similarities found. Try another book.")
        else:
            for col, book in zip(cols, results):
                with col:
                    st.image(book["Image-URL-M"], use_container_width=True)
                    st.markdown(f"**{book['Book-Title']}**")
                    st.caption(book["Book-Author"])


# USER-BASED

elif page == "👤 User-based":
    st.title("👤 User-based Recommendation")

    user_id = st.selectbox("Select User ID", sorted(user_map.keys()))

    if st.button("👤 Recommend"):
        books, _ = recommend_books_for_user(user_id)

        num_cols = max(1, min(len(books), 5))
        cols = st.columns(num_cols)

        if not books:
            cols[0].info("No recommendations available for this user.")
        else:
            for col, book in zip(cols, books):
                with col:
                    st.image(book["Image-URL-M"], use_container_width=True)
                    st.markdown(f"**{book['Book-Title']}**")
                    st.caption(book["Book-Author"])

# USER ANALYTICS

elif page == "📊 User Analytics":
    st.title("📊 User Analytics Dashboard")

    selected_user = st.selectbox("Select User ID", sorted(user_map.keys()))

    books, scores = recommend_books_for_user(selected_user, n=10)

    if scores:
        col1, col2 = st.columns(2)
        col1.metric("User ID", selected_user)
        col2.metric("Recommendations Analysed", len(scores))

        # Rating Trend
        fig1, ax1 = plt.subplots()
        ax1.plot(scores, marker='o')
        ax1.set_title("Predicted Preference Trend")
        ax1.set_xlabel("Recommendation Index")
        ax1.set_ylabel("Score")
        st.pyplot(fig1)

        # Score Distribution
        fig2, ax2 = plt.subplots()
        ax2.hist(scores, bins=10)
        ax2.set_title("Recommendation Score Distribution")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        # Similar Users
        similar_users = find_similar_users(selected_user)

        st.subheader("👥 Similar Users")
        st.write(similar_users)

        fig3, ax3 = plt.subplots()
        ax3.bar(range(len(similar_users)), [1]*len(similar_users))
        ax3.set_title("Top Similar Users")
        ax3.set_xlabel("User Index")
        ax3.set_ylabel("Similarity")
        st.pyplot(fig3)
