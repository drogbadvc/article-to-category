import requests
import streamlit as st
import validators
import pandas as pd


def is_valid_url(url):
    return validators.url(url)


def check_url_accessibility(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data
def panda_frame(table_data):
    return pd.DataFrame(table_data, columns=["URL", "Category"])


def classify_page_cat():
    st.title("Classification of articles in categories")
    form = st.form(key='my_form')
    url_input = form.text_area("Enter one or more URLs (one URL per line)")
    option = form.radio("Choose an option:", ("Title", "ExtractedText"))
    word_list_input = form.text_input("Enter a list of categories (separated by commas)")
    submit_button = form.form_submit_button(label='Confirm')

    if submit_button:
        with st.spinner(''):
            url_list = url_input.splitlines()
            if not url_list or not word_list_input or not option:
                st.warning("Please fill in all fields and select an option.")
            else:
                invalid_urls = [url for url in url_list if not is_valid_url(url)]
                if invalid_urls:
                    st.warning(f"These URLs are invalid : {invalid_urls}")
                    return
                inaccessible_urls = [url for url in url_list if not check_url_accessibility(url)]
                if inaccessible_urls:
                    st.warning(f"These URLs are not accessible : {inaccessible_urls}")
                    return
                word_list = [word.strip() for word in word_list_input.split(",")]
                if len(word_list) < 2:
                    st.warning("Please enter at least two keywords separated by commas.")
                elif len(word_list) != len(set(word_list)):
                    st.warning("Please enter unique, unduplicated keywords.")
                else:
                    data = {
                        'url_input': url_list,
                        'option': option,
                        'word_list': word_list
                    }
                    response = requests.post('http://localhost:8000/classify/', json=data)
                    if response.status_code == 200:
                        result = response.json()

                        table_data = []

                        for url, classification in result.items():
                            if "Error" not in classification:
                                highest_scoring_category = max(classification["scores"])
                                index = classification["scores"].index(highest_scoring_category)
                                category = classification["labels"][index]

                                table_data.append([url, category])

                        df = panda_frame(table_data)

                        st.dataframe(df, use_container_width=True)

                        csv = df.to_csv(index=False).encode()
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name='classification_results.csv',
                            mime='text/csv',
                        )
                    else:
                        st.error("An error has occurred during classification. Error code : {}".format(
                            response.status_code))


if __name__ == "__main__":
    classify_page_cat()
