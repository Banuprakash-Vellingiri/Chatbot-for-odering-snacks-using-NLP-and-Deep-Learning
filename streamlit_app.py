#Chatbot web application using streamlit 
#--------------------------------------------------------------------------------------------------------------
#Importing Dependencies
import streamlit as st
import json

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from datetime import datetime

import spacy
from tensorflow.keras.models import load_model
import pickle
#--------------------------------------------------------------------------------------------------------------
#streamlit environment
#Page Layout
st.set_page_config (
                    page_title="Banu Snacks Chatbot",
                    page_icon= "♨️",  
                    # layout="wide",  
                    initial_sidebar_state="expanded",  
                   )
#Heading 
st.markdown("# ♨️ :orange[Banu Snacks Chatbot]")
st.write("###### 😋 Order Your Favourite Snacks")
st.image("chatbot_logo.png")
st.markdown("*"*100)
#--------------------------------------------------------------------------------------------------------------
#Loading the saved models
#DL model
dl_model= load_model("dl_model.h5")
#tf-idf model
with open('tfidf_model.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
#===================================================================================================================
#Loading json file(intents) from directory
with open("intents.json", "r", encoding="utf-8") as file:
    intents= json.load(file)
#===================================================================================================================
def preprocess_text(text):
    sentences = sent_tokenize(text)
    filtered_tokens = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalnum()]
        filtered_tokens.extend(lemmatized_tokens)
    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text

#===================================================================================================================
#Function for recognizing the Labels.It utilizizes custom trained spacy model.
def order(text):
    #---------------------------------------------------------------------------------------------------------------
    ordered_snacks_list = []
    ordered_snack_quantity_list = []
    #---------------------------------------------------------------------------------------------------------------
    #Loading the custom model
    custom_ner_model=spacy.load("custom_ner_model_2")
    #---------------------------------------------------------------------------------------------------------------
    text=text.lower()
    doc = custom_ner_model(text)
    #---------------------------------------------------------------------------------------------------------------
    for ent in doc.ents:
        if ent.label_ == "SNACKS":
            ordered_snacks_list.append(ent.text)
        if ent.label_ == "NUMBERS":
            ordered_snack_quantity_list.append(ent.text)
    return ordered_snacks_list ,ordered_snack_quantity_list
#===================================================================================================================
#List for storing orders
order_list=[]
#-------------------------------------------------------------------------------------------
#Function for text input andd to return output
def input_chat(input_text):
              input_text=input_text.lower()
              #------------------------------------------------------------------------------
              #Text Preprocessing
              processed_chat=preprocess_text(input_text)
              #Converting to vector
              processed_chat_vector=tfidf_vectorizer.transform([processed_chat])
              #------------------------------------------------------------------------------
              #Prediction
              predicted_probability=dl_model.predict(processed_chat_vector.toarray())
              #------------------------------------------------------------------------------
              output_label_data={
                            0: 'Greetings',
                            1: 'Well_Being_Enquiry',
                            2: 'About_Me',
                            3: 'Snacks_Recommendations',
                            4: 'Order_Confirmation_no',
                            5: 'Order_Confirmation_yes',
                            6: 'Ordering_Intent_With_Quantity',
                            7: 'Order_Without_Quantity',
                            8: 'Final_Words',
                            9: "Not_Available"
                            }
              #------------------------------------------------------------------------------
              predicted_tag=output_label_data[predicted_probability.argmax()]
              #-----------------------------------------------------------------------------
              #Response
              for dic in intents["intents"]:
                       if dic["tag"]==predicted_tag:
                            output_response=np.random.choice(dic["responses"])
                            if predicted_tag=="Snacks_Recommendations":
                                   menu_image=Image.open("menu.jpg")
                                   #display(menu_image)
                                   st.image(menu_image)
                                   return output_response
                            if predicted_tag=="Not_Available":
                                   menu_image=Image.open("menu.jpg")
                                   #display(menu_image)
                                   st.image(menu_image)
                                   return output_response
                            if predicted_tag=="Ordering_Intent_With_Quantity":
                                   #----------------------------------------------------------------------------- 
                                   ordered_snacks_list ,ordered_snack_quantity_list=order(input_text)
                                   #-----------------------------------------------------------------------------
                                   order_list.append(input_text)
                                   #-----------------------------------------------------------------------------
                                   total_price=[]  
                                   #----------------------------------------------------------------------------- 
                                   for snack,quantity in zip(ordered_snacks_list ,ordered_snack_quantity_list):
                                       price_list =    {
                                                        "bajji": 8,
                                                        "bajjis": 8,
                                                        "bread omelette": 55,
                                                        "bread omelettes": 55,
                                                        "chicken puff": 25,
                                                        "chicken puffs": 25,
                                                        "chicken roll": 30,
                                                        "chicken rolls": 30,
                                                        "curd vada": 25,
                                                        "curd vadas": 25,
                                                        "egg puff": 20,
                                                        "egg puffs": 20,
                                                        "gobi chilli": 45,
                                                        "gobi chillis": 45,
                                                        "masala bonda": 12,
                                                        "masala bondas": 12,
                                                        "masala vada": 8,
                                                        "masala vadas": 8,
                                                        "mushroom chilli": 60,
                                                        "mushroom chillis": 60,
                                                        "onion bonda": 8,
                                                        "onion bondas": 8,
                                                        "onion samosa": 10,
                                                        "onion samosas": 10,
                                                        "potato samosa": 10,
                                                        "potato samosas": 10,
                                                        "rusk": 7,
                                                        "rusks": 7,
                                                        "salt biscuit": 6,
                                                        "salt biscuits": 6,
                                                        "sambar vada": 25,
                                                        "sambar vadas": 25,
                                                        "veg puff": 15,
                                                        "veg puffs": 15,
                                                        "veg roll": 23,
                                                        "veg rolls": 23
                                                       }
                                       word_to_digit_mapping = {
                                                        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 
                                                        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                                                        'nineteen': 19, 'twenty': 20, 'twenty one': 21, 'twenty two': 22, 'twenty three': 23, 'twenty four': 24, 'twenty five': 25, 
                                                        'twenty six': 26, 'twenty seven': 27, 'twenty eight': 28, 'twenty nine': 29, 'thirty': 30, 'thirty one': 31, 'thirty two': 32, 'thirty three': 33, 'thirty four': 34, 'thirty five': 35, 'thirty six': 36, 'thirty seven': 37, 'thirty eight': 38, 'thirty nine': 39, 'forty': 40, 'forty one': 41, 'forty two': 42, 'forty three': 43, 'forty four': 44, 'forty five': 45, 'forty six': 46, 'forty seven': 47, 'forty eight': 48, 'forty nine': 49, 'fifty': 50, 'fifty one': 51, 'fifty two': 52, 'fifty three': 53, 'fifty four': 54, 'fifty five': 55, 'fifty six': 56, 'fifty seven': 57, 'fifty eight': 58, 'fifty nine': 59, 'sixty': 60,
                                                        'sixty one': 61, 'sixty two': 62, 'sixty three': 63, 'sixty four': 64, 'sixty five': 65, 'sixty six': 66, 'sixty seven': 67, 'sixty eight': 68, 'sixty nine': 69, 'seventy': 70, 'seventy one': 71, 'seventy two': 72, 'seventy three': 73, 'seventy four': 74, 
                                                        'seventy five': 75, 'seventy six': 76, 'seventy seven': 77, 'seventy eight': 78, 'seventy nine': 79, 'eighty': 80, 'eighty one': 81, 'eighty two': 82, 'eighty three': 83, 'eighty four': 84, 'eighty five': 85, 'eighty six': 86, 'eighty seven': 87, 'eighty eight': 88, 'eighty nine': 89, 'ninety': 90, 'ninety one': 91, 'ninety two': 92, 'ninety three': 93, 'ninety four': 94, 'ninety five': 95, 'ninety six': 96, 'ninety seven': 97, 'ninety eight': 98, 'ninety nine': 99, 'one hundred': 100
                                                        }
                                       if quantity.isdigit():                                                   
                                           total_price.append( price_list[snack]*int(quantity)) 
                                       else:                                                   
                                          total_price.append(price_list[snack]*word_to_digit_mapping[quantity])
                                   #-----------------------------------------------------------------------
                                   bill_df=pd.DataFrame({"SI":[i for i in range(1,len(ordered_snacks_list)+1)],"Snacks":ordered_snacks_list,"Price (₹)":[price_list[snack]for snack in (ordered_snacks_list) ],"Quantity":[int(word_to_digit_mapping[i]) if i in word_to_digit_mapping else i for i in ordered_snack_quantity_list],"Total Price (₹)":total_price},index=None) 
                                   #-----------------------------------------------------------------------
                                   total_bill=bill_df["Total Price (₹)"].sum()
                                   #--------------------------------------------------------------------------------   
                                   st.markdown( "##### Your Snacks Wishlist ⤵️" )
                                   st.table(bill_df)
                                   st.markdown(f'##### :orange[Net Amount :] ₹ {float(total_bill)}')          
                                   return "Would you like to confirm your order ?"
                            
                            if predicted_tag=="Order_Confirmation_yes":
                               with open("order_list.json", 'r') as file:
                                    loaded_order_list = json.load(file)
                               if len(loaded_order_list)>=1: 
                                   ordered_snacks_list ,ordered_snack_quantity_list=order(loaded_order_list[-1])
                                   total_price=[]   
                                   for snack,quantity in zip(ordered_snacks_list ,ordered_snack_quantity_list):
                                       price_list =    {
                                                        "bajji": 8,
                                                        "bajjis": 8,
                                                        "bread omelette": 55,
                                                        "bread omelettes": 55,
                                                        "chicken puff": 25,
                                                        "chicken puffs": 25,
                                                        "chicken roll": 30,
                                                        "chicken rolls": 30,
                                                        "curd vada": 25,
                                                        "curd vadas": 25,
                                                        "egg puff": 20,
                                                        "egg puffs": 20,
                                                        "gobi chilli": 45,
                                                        "gobi chillis": 45,
                                                        "masala bonda": 12,
                                                        "masala bondas": 12,
                                                        "masala vada": 8,
                                                        "masala vadas": 8,
                                                        "mushroom chilli": 60,
                                                        "mushroom chillis": 60,
                                                        "onion bonda": 8,
                                                        "onion bondas": 8,
                                                        "onion samosa": 10,
                                                        "onion samosas": 10,
                                                        "potato samosa": 10,
                                                        "potato samosas": 10,
                                                        "rusk": 7,
                                                        "rusks": 7,
                                                        "salt biscuit": 6,
                                                        "salt biscuits": 6,
                                                        "sambar vada": 25,
                                                        "sambar vadas": 25,
                                                        "veg puff": 15,
                                                        "veg puffs": 15,
                                                        "veg roll": 23,
                                                        "veg rolls": 23
                                                       }
                                       word_to_digit_mapping = {
                                                        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 
                                                        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                                                        'nineteen': 19, 'twenty': 20, 'twenty one': 21, 'twenty two': 22, 'twenty three': 23, 'twenty four': 24, 'twenty five': 25, 
                                                        'twenty six': 26, 'twenty seven': 27, 'twenty eight': 28, 'twenty nine': 29, 'thirty': 30, 'thirty one': 31, 'thirty two': 32, 'thirty three': 33, 'thirty four': 34, 'thirty five': 35, 'thirty six': 36, 'thirty seven': 37, 'thirty eight': 38, 'thirty nine': 39, 'forty': 40, 'forty one': 41, 'forty two': 42, 'forty three': 43, 'forty four': 44, 'forty five': 45, 'forty six': 46, 'forty seven': 47, 'forty eight': 48, 'forty nine': 49, 'fifty': 50, 'fifty one': 51, 'fifty two': 52, 'fifty three': 53, 'fifty four': 54, 'fifty five': 55, 'fifty six': 56, 'fifty seven': 57, 'fifty eight': 58, 'fifty nine': 59, 'sixty': 60,
                                                        'sixty one': 61, 'sixty two': 62, 'sixty three': 63, 'sixty four': 64, 'sixty five': 65, 'sixty six': 66, 'sixty seven': 67, 'sixty eight': 68, 'sixty nine': 69, 'seventy': 70, 'seventy one': 71, 'seventy two': 72, 'seventy three': 73, 'seventy four': 74, 
                                                        'seventy five': 75, 'seventy six': 76, 'seventy seven': 77, 'seventy eight': 78, 'seventy nine': 79, 'eighty': 80, 'eighty one': 81, 'eighty two': 82, 'eighty three': 83, 'eighty four': 84, 'eighty five': 85, 'eighty six': 86, 'eighty seven': 87, 'eighty eight': 88, 'eighty nine': 89, 'ninety': 90, 'ninety one': 91, 'ninety two': 92, 'ninety three': 93, 'ninety four': 94, 'ninety five': 95, 'ninety six': 96, 'ninety seven': 97, 'ninety eight': 98, 'ninety nine': 99, 'one hundred': 100
                                                        }
                                       if quantity.isdigit():                                                   
                                           total_price.append( price_list[snack]*int(quantity)) 
                                       else:                                                   
                                          total_price.append(price_list[snack]*word_to_digit_mapping[quantity])
                                   #-----------------------------------------------------------------------
                                   bill_df=pd.DataFrame({"SI":[i for i in range(1,len(ordered_snacks_list)+1)],"Snacks":ordered_snacks_list,"Price (₹)":[price_list[snack]for snack in (ordered_snacks_list) ],"Quantity":[int(word_to_digit_mapping[i]) if i in word_to_digit_mapping else i for i in ordered_snack_quantity_list],"Total Price (₹)":total_price},index=None) 
                                   #-----------------------------------------------------------------------
                                   total_bill=bill_df["Total Price (₹)"].sum()
                                   #-----------------------------------------------------------------------
                                   order_id=np.random.randint(1000,10000)
                                   #----------------------------------------------------------------------
                                   current_datetime =datetime.now()
                                   formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                                   
                                   #----------------------------------------------------------------------
                                   st.markdown("### ♨️ :orange[Banu Snacks]")
                                   st.markdown(f'##### Date | Time : {formatted_datetime}')
                                   st.table(bill_df)
                                   st.markdown(f'##### :orange[Order ID :]     #{order_id}')
                                   st.markdown(f'##### :orange[Net Amount :] ₹ {float(total_bill)}')
                                   st.markdown("##### Thanks for Ordering! Spread Positivity ❤️")
                                   return "Your order is confirmed Thank you! Visit Again."
                               else:
                                  return "Kindly specify your order!"
                                   
                            else:
                                   return output_response
                            
#===================================================================================================================
# #User input
# text_input=st.text_input("Order your snacks")

# #----------------------------------------------------------------------------------------------
# #Calling response function to get response from the chatbot
# if text_input:
#     try:
#         response=input_chat(text_input)
#         print(response)
#         st.write(f"#### {response}")
#     except Exception:
#         print("Sorry! Invalid Entry \n I can't able to understand,try again!.")  
#----------------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
#----------------------------------------------------------------------------------------------
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"]=="user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if message["role"]=="assistant":
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 
#----------------------------------------------------------------------------------------------
request = st.chat_input("Say something")
if request  :
    # Display user message in chat message container
    with st.chat_message(name="user"):
        st.markdown(request)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": request})
    try:
        bot_response=input_chat(request)
        response = f'##### {bot_response}'
    except Exception:
        response="Sorry! Invalid Entry \n I can't able to understand,try again!." 
# Display assistant response in chat message container
    with st.chat_message(name="assistant"):
        st.write(response)
# Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 
#----------------------------------------------------------------------------------------------
#Saving the ordered snacks text locally      
if len(order_list)>=1:
    with open("order_list.json", 'w') as file:
      json.dump(order_list, file)
#---------------------------------------------------------------------------------------------- 
