import streamlit as st
from movie_Recommendation import recommend  # Import your function from the Python script

# Main function to run the Streamlit app
def main():
    # Set title and sidebar
    st.title("Movie Recommender System")
    # st.sidebar.title("Options")

    # Get user input
    user_input = st.text_input("Enter user preferences (e.g., genre, actor, director):")

    # Check if user input is provided
    if user_input:
        # Get recommendations using the imported function with user input
        recommendations = recommend(user_input)

        # Display recommendations
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("- " + movie)
    else:
        st.write("Please enter your preferences.")

# Run the main function
if __name__ == "__main__":
    main()
