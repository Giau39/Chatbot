#import all the required libraries of python
import random
import string
import json
import os
import numpy as np 
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
from nltk.stem.snowball import SnowballStemmer
sb_stemmer = SnowballStemmer('english')
nltk.download('words')
from nltk.corpus import words
correct_words = words.words()

#INTENT IDENTIFICATION FEATURE
# Loading json data for the intent identification
with open('intents.json') as file:
    data = json.loads(file.read())

#for predicting the intent
tags=[] #tags are the intent of the data in which user resonses are predicted
uniquetags=[]
patterns=[] #patterns are the possible user responses
responses=[] #responses are the random reponse chatbot should give for the predicted intent

words =[] #vocabulary of intent classification lemitized in responses
for element in data['intents']:
    uniquetags.append(element['tag'])    
    for patternSentence in element['patterns']:# patterns and the tags are created for the classification performed on later stage
        tags.append(element['tag'])
        patterns.append(patternSentence)    
        
    for patternLines in element['patterns']:#patterns are lemmitized and added for better performance
         for wordInLine in patternLines.split():
            lemm_word=lemmatizer.lemmatize(wordInLine.lower())
            if lemm_word not in words:
                words.append(lemm_word)                
    
    responses.append(element['responses'])#responses added from json file to array
    
# Intent classification data is splitted into train and test dataset for training the model
x_train,x_test,y_train,y_test= train_test_split(patterns,tags,stratify=tags,test_size=0.2,random_state=42)#stratify=tags,
count_vect = CountVectorizer(stop_words=stopwords.words('english'))
x_train_counts= count_vect.fit_transform(x_train) #x_train

#training the model
tfidf_transformer = TfidfTransformer(use_idf=True,sublinear_tf=True).fit(x_train_counts)
x_train_tf= tfidf_transformer.transform(x_train_counts)



x_new_counts = count_vect.transform(x_test)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

#Support Vector Machine
from sklearn import svm
svcclf = svm.SVC()
svcclf.fit(x_train_tf, y_train)
svcpredicted= svcclf.predict(x_new_tfidf)

#general knowledge section #tokenizing and kemmitizing the sentences   
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#predict the intent for given user input
def predict_class(sentence):
    doc=[]
    doc.append(sentence)
    x_new_counts = count_vect.transform(doc) # x_test = arrayOfWords
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)
    predicted= svcclf.predict(x_new_tfidf)
    return predicted

#get random response of predicted tag 
def getResponse(ints,intents_json):
    tag= ints
    result=""
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        # lấy ký hiệu đầu tiên để gán 
        if(i['tag']== tag[0]):
            result = random.choice(i['responses'])
            break

    return result



#chatbot work here is to predict the tag and get the response for predicted tag
def chatbot_response(text):
    ints = predict_class(text)
    res = getResponse(ints, data)
    return res

#get teh name of the person from the sentence #FOR IDENTITY MANAGEMENT FEATURE
def getName(sentence):
    postArray=" "
    post = pos_tag(sentence.split())
    for postArray in post[::-1]:
        if(postArray[1]=='NN' or postArray[1]=='NNP' or postArray[1]=='JJ'):
            return (postArray[0]).capitalize()

#load the dataset for QUESTION ANSWERING FEATURE
import pandas as pd;
df=pd.read_csv("data.csv")
df.columns=["Question_ID","Question","Answer","Document"]
global qa_documents
qa_documents={}
global topiclist
topiclist=[]
topic_documents={}

for index,row in df.iterrows():
        if pd.isnull(df.values).any(1)[index]:
            row["Answer"]="I dont know. I will ask teacher to let u know." #give this response if the data is not available
            
        if row["Question"] in qa_documents:
            qa_documents[row["Question"]]= qa_documents[row["Question"]] + " " + row["Answer"]#merge the answers for same question
        else:
            qa_documents[row["Question"]]= row["Answer"]
        
        if row["Document"] not in topiclist and row["Document"] is not None:
            topiclist.append(row["Document"])#creating topic list
            
        topic_documents[row["Question"]]=row["Document"]

#tokenize sentence and remove stopwords if required
def clean_sentence(sentence, stopwordsAbsent=False):
    tokenizer=nltk.RegexpTokenizer(r"\w+")
    text_tokens = tokenizer.tokenize(sentence)
    tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words('english')] #convert to lowercase
    if stopwordsAbsent:
        sentence = (" ").join(tokens_without_sw)
    return sentence

#tokenize each answers
def get_cleaned_sentence(df, stopwordsAbsent=True):

    cleaned_sentences=[]
    for index,row in df.iterrows():
        cleaned=clean_sentence(row["Question"],stopwordsAbsent)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

#tokenization
tokenizer=nltk.RegexpTokenizer(r"\w+")
tok_documents={}
for title in qa_documents:
    tok_documents[title] = tokenizer.tokenize(title)

#remove stopwords and normalise casing
english_stopwords = stopwords.words('english')
filtered_documents={}
for title in tok_documents:
    filtered_documents[title] = [word.lower() for word in tok_documents[title] if word.lower() not in english_stopwords]   

#do the stemming
stemmed_documents ={}
for title in filtered_documents:
    stemmed_documents[title]= [sb_stemmer.stem(word) for word in filtered_documents[title]]

#create the vocabulary
vocabulary =[]
for title in stemmed_documents:
    for stem in stemmed_documents[title]:
        if stem not in vocabulary:
            vocabulary.append(stem)

#create a bag of words
def doc2bow(documents,related_vocab):
    bow ={}
    for title in documents:
        bow[title] = np.zeros(len(related_vocab))# list of same size of vocabulary
        for stem in documents[title]:
            try:
                index= related_vocab.index(stem)
                bow[title][index] += 1
            except ValueError:
                continue     
    return bow
BOW=doc2bow(stemmed_documents,vocabulary)#BOW of each answers in document

#cosine similarity between vector documents
def response(question_orig):
    #word embedding #to do
    question= clean_sentence(question_orig,stopwordsAbsent=True)
    questionDocument={}
    stemmed_question_doc={}
    listOfCorrectWords=[]
    questionDocument[0]=question.split()
    stemmed_question_doc[question]= [sb_stemmer.stem(word) for word in questionDocument[0]]
    question_vector=doc2bow(stemmed_question_doc,vocabulary)[question]
    getTheAnswer(question,question_vector,BOW,qa_documents,stemmed_documents)

#this code is only for testing and creating report content of this project
def manhattan(vector_1, vector_2):
    diff = abs(vector_1 - vector_2)
    return diff.sum()

def jaccard(x,y):
    x = np.asarray(x, np.bool) # Not necessary, if you keep your data
    y = np.asarray(y, np.bool) # in a boolean array already!
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

import warnings
warnings.filterwarnings("ignore")  

#Get the answers as a response from an excel sheet
from scipy import spatial
def getTheAnswer(question,question_vector, sentence_vector,qa_documents,sentences):
    max_sim=-1
    index_sim=-1
    sim=0.0001
    if(np.sum(question_vector)==0):
        print("Sorry I am unable to find it")
        return
    maxSimIndexArray=[]
    
    for index in sentence_vector.keys():
        try:
            sim=1-spatial.distance.cosine(question_vector,sentence_vector[index])
        except:
            continue
        if(sim>max_sim):
            max_sim=sim
            index_sim=index
            maxSimIndexArray=[]
            maxSimIndexArray.append(index)
        elif(sim==max_sim):
            maxSimIndexArray.append(index)
    
    answer=""
    if(len(maxSimIndexArray)>1):
        print("I am sorry but but it is not clear to me. Are u asking anything like the following?(Please select one question from the following):")
        num=1
        for index in maxSimIndexArray :
            print(num,". ",index)
            num=num+1
        print(num+1,". None of these")
        user_selection = input()
        if(user_selection==num+1):
            print("Sorry I dont have any info related to this in my database.")
            return 
        if(not (user_selection).isdigit() or int(user_selection)>=num): 
            print("Sorry but I am unable to understand.")
            return
        index_sim=maxSimIndexArray[int(user_selection)-1]
    
    answer= qa_documents[index_sim]    
    
    if(index_sim==-1):
        print("Sorry I am unable to find anything related to this.Please try again.")
    else:    
        print(answer)     


#game section
df_game=pd.read_excel(r'character.xlsx')
game_documents={}
for index,row in df_game.iterrows():
    cdf=""
    for i in range (1,12):
        if pd.isnull(df_game.iloc[index][i]):
            continue
        cdf = cdf + " " + str(df_game.iloc[index][i])
    game_documents[row["Name"]] = cdf

cleaned_details=""    
chatBot=""
#Chatbot is considering any random character name
chatBot=random.choice(list(game_documents.items()))[0]
#print(chatBot)

#clean the information we are getting for the character
def get_clean_details(chatBot,documents):
    details=documents[chatBot]
    cleaned_details=clean_sentence(details,True)
    return cleaned_details

cleaned_details= get_clean_details(chatBot,game_documents)
#predict_class function will give whether the word is present or not in the text
def predict_character(user_response):
    cleaned_question=clean_sentence(user_response,True)
    yesOrNo= getYesOrNO(cleaned_details,cleaned_question)
    return yesOrNo
#creating vocabulary
game_vocabulary=[]
def createVocabulary(cleaned_passage):
    #tokenization
    tokenizer=nltk.RegexpTokenizer(r"\w+")
    tok_documents={}
    related_vocabulary=[]
    tok_documents = tokenizer.tokenize(cleaned_passage)
    for stem in tok_documents:
        if (stem not in related_vocabulary):
            related_vocabulary.append(stem)
    return related_vocabulary
game_vocabulary=createVocabulary(cleaned_details)

#creating bag of words on the basis of vocabulary
def sentence2bow(sentence,relevant_vocabulary):
    bowVector = np.zeros(len(relevant_vocabulary)) # list of same size of vocabulary
    array = []
    array = sentence.split()
    for stem in array:
        try:
            index= relevant_vocabulary.index(stem)
            bowVector[index] += 1
        except ValueError:
            continue     
    return bowVector

bowOfPassage=[]
bowOfPassage=sentence2bow(cleaned_details,game_vocabulary)

#get the yes or no answers from chatbot
def getYesOrNO(cleaned_details,cleaned_question):
    max_sim=-1
    index_sim=-1
    bow=sentence2bow(cleaned_question,game_vocabulary)
    if(not bow.any()):
        sim=-1
    try:
        sim=1-spatial.distance.cosine(bow,bowOfPassage)
    except ValueError:
        sim=-1
    if(sim>max_sim):
        return True 
    return False

#hint function will generate the hint in the blank format (Harry Potter ->> ----- ------)
def hint(show):
    #print(chatBot)
    splitPersonName=chatBot.split()
    if(show):
        for word in splitPersonName:
            for i in range (0,len(word)):
                print("_",end =" ")
            print(" ",end =" ")
        print(" ")


#if user wants to play again then reset will reset all the values
def reset():
    global chatBot
    chatBot=random.choice(list(game_documents.items()))[0]
    global game_vocabulary
    game_vocabulary=[]
    global cleaned_details
    cleaned_details= get_clean_details(chatBot,game_documents)
    game_vocabulary=createVocabulary(cleaned_details)
    global bowOfPassage
    bowOfPassage=[]
    bowOfPassage=sentence2bow(cleaned_details,game_vocabulary)

def gameCall():
    gameflag=1
    count=0
    print("--------------------------------------------------------------------------------------")
    print("Max:Hi there! Lets check if u can guess the famous characters from 'Harry Potter' series")
    print("Rules:")
    print("1. You can ask me questions related to that person. e.g. Is he/she male or female? Is she a student? Is he a professor?")
    print("2. I will say only Yes or No.")
    print("3. You have to guess the person in 10 yes or no questions.")
    print("Ok as you are a beginer I will tell you if you are close or very close also...just for you my friend")
    print("If you need a hint just say 'Hint'. I will give you one clue. There is only one clue available.")
    #print("Your score will be decided on your number of questions asked and on the hint taken it or not.")
    print("I hope you are clear with the rules. So lets begin. I have decided one person in my mind. Guess Who?")
    print("--------------------------------------------------------------------------------------")
    while gameflag==1:
        hint(False)
        print(name,": ",end="")
        user_response=input()
        user_response = spellCorrector(user_response)
        count=count+1
        if((user_response).lower()=="hint"): 
            hint(True)
        elif(predict_class(user_response)=="quit" or count==10 or user_response.lower=="i quit"):
            if count==5 : print("Max: You lost your 10 chances.")
            print("Max: The character is ",chatBot)
            print(game_documents[chatBot])
            print("Max:Do you wanna play it again?")
            print(name,": ",end="")
            repeat=input()
            gameflag=0
            if(repeat.lower()=='yes'):
                reset()
                gameflag=1
            else:
                print("What do you wanna do next?")
        elif((user_response).lower()==(chatBot).lower()):
            print("Max:Congratulations!!! You guessed it correctly. You are so smart. Good game.")
            print("Do you wanna play it again?")
            print(name,": ",end="")
            repeat=input()
            gameflag=0
            if(repeat.lower()=='yes'):
                reset()
                gameflag=1
        elif(predict_class(user_response)!='bye'):
            similarity=predict_character(user_response)
            if(similarity):
                print("Max:Yes. You are close.")
            else:
                print("Max:No")
        else:
            gameflag=0


#transaction feature: For booking a test
testBooking={"date":["Ok great. For what date and start time I should book the test? ","Thats awesome. On what date and start time u wanna appear for this test? "],
             "bookdate":["Ok great. I have booked it for u on ","Thats awesome. Now u have test on "],
             
             "Topic":[ "Ok great. What should be the topics? ","Done. Which topics u wanna cover in this test?"],
             "bookTopic":[ "Ok great. I am adding this/these topic/s - ","Done. I have added these topics -"],
             
             "Marks":[ "What marks u wanna give for each question? ","How many marks would u like to give to every question? "],
            "bookMarks":[ "For confirmation I have added marking per question as ","Done. Your marks for each question for this test will be "],
     
             "no of questions":[ "How many questions u wanna add? ","What should be the number of questions to add in this test? "],
             "book no of questions":[ "Done. There will be total  ","Nice. This will be a good test with "],
             
             "total time":["For how many mins?"],
             "book total time":["Great ur test will be of "],
            
             "Wish luck":["All the best","Best of luck","Do well in exam,"]         
            }
#get the topics name from the user input topic list
def getTopics(cleanData):
    topicsInTest=[]
    topics_vocabulary=[]
    for topic in topiclist:
        for words in createVocabulary(topic):
            if(words.lower() not in topics_vocabulary):
                topics_vocabulary.append(words.lower())
    max_sim=0
    for topic in topiclist:
        bowOfRequestedTopic=sentence2bow(cleanData,topics_vocabulary)
        bowOfTopic=sentence2bow(topic.lower(),topics_vocabulary)
        if(not bowOfRequestedTopic.any()):
            continue
        try:
            sim=1-spatial.distance.cosine(bowOfRequestedTopic,bowOfTopic)
        except ValueError:
            continue     
        if(sim>max_sim):
            topicsInTest.append(topic)
    return topicsInTest

#print the confirmed data
import dateutil
def infoFromUserResponse(sentences,intent):
    cleanData=clean_sentence(sentences,True)
    numberFromData=0
    for words in sentences.split(): 
        try:
            if words.isdigit(): 
                numberFromData=words
        except:
            print("Transaction is closed.")
            continue
    if intent == 'bookdate': 
        try:
            d=dateutil.parser.parse(sentences, fuzzy=True)
            print(d)
        except:
            print("Transaction is closed.")
    elif intent == 'bookTopic':
        topicListForTest=getTopics(cleanData)
        for topic in topicListForTest:
            print(" ",topic)
    elif intent == 'bookMarks':
        print(numberFromData + " marks")
    elif intent == 'book no of questions':
        print(numberFromData + " questions")
    elif intent == 'book total time':
        print(numberFromData + " mins")
        
#store the transactional data of booking a test    
def storeData():
    for index,key in enumerate(testBooking):
        if(index%2 ==0 ): 
            print("Max: ",end="")
            print(random.choice(testBooking[key]))
            print(name,": ",end="")
            user_response=input()
            if(not has_numbers(user_response)):
                user_response = spellCorrector(user_response)
        else: 
            print("Max: ",end="")
            print(random.choice(testBooking[key]),end="")
            try:
                infoFromUserResponse(user_response,key)
            except:
                print("Somethign is wrong.")
                print("Transaction closed.")
                return False
        if(key=="Wish luck"):
            print("All the details are confirmed. Your test is booked.",end=" ")
            print(random.choice(testBooking[key]),name)
            print("Max: What would you like to do now?")
        if(user_response=="bye"):
            return False
    return False

testmins=0
noOfQuestions=0
topics=[]
marks=0
testBookingCompleted=False

#to ask questions to user
def checkTheSimilarityOFAnswers(user_answer,question):
    cleanSentences=clean_sentence(user_answer, stopwordsAbsent=True)
    cleaned_details= get_clean_details(question,qa_documents)
    answer_vocabulary=[]
    answer_vocabulary=createVocabulary(cleaned_details)
    bowOfPassage=[]
    bowOfPassage=sentence2bow(cleaned_details,answer_vocabulary)
    
    max_sim=-1
    index_sim=-1
    bow=sentence2bow(cleanSentences,answer_vocabulary)
    if(not bow.any()):
        return 0
    try:
        sim=1-spatial.distance.cosine(bow,bowOfPassage)
    except:
        sim=0   
    return sim

def checkTheSimilarityOFQuestion(user_answer,question):
    cleanSentences=clean_sentence(user_answer, stopwordsAbsent=True)
    cleaned_details= clean_sentence(question, stopwordsAbsent=True)
    question_vocabulary=[]
    question_vocabulary=createVocabulary(cleaned_details)
    bowOfPassage=[]
    bowOfPassage=sentence2bow(cleaned_details,question_vocabulary)
    bow=sentence2bow(cleanSentences,question_vocabulary)
    if(not bow.any()):
        return 0
    try:
        sim=1-spatial.distance.cosine(bow,bowOfPassage)
    except:
        sim=0   
    return sim

#show results
askedQuestions={}
def showResults(askedQuestions):
    if askedQuestions: 
        print("----------------------------------------------------------------------------------")
        print("Report:")
        trunkTopicName=[]
        trunkQuestion=[]
        score=[]
        for question in askedQuestions:
            trunkTopicName.append((topic_documents[question][:30] + '...') if len(topic_documents[question]) > 30 else topic_documents[question][:30])
            trunkQuestion.append((question[:20] + '...') if len(question) > 20 else question[:20])
            score.append(askedQuestions[question])
            result=pd.DataFrame(list(zip(trunkQuestion, trunkTopicName, score)))
        result.columns=['Question', 'Topic','Score']
        result.index = range(1,len(trunkQuestion)+1)
        print(result)
        if np.mean(score)<50:
            print("----------------------------------------------------------------------------------")
            print("Max: Dont worry",name,".We will practice and practice makes a person perfect." )

#checking user response         
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

#correct the spells excluding nouns
def spellCorrector(sentence):
    correctedWords=[]
    for word in list(sentence.split()):
        if nltk.pos_tag([word])[0][1]=='NNP' or nltk.pos_tag([word])[0][1]=='NN':
            correctedWords.append(word)
            continue
        temp = [(nltk.edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
        correctWord=sorted(temp, key = lambda val:val[0])[0][1]
        correctedWords.append(correctWord)
    correctedSentence=(" ").join(correctedWords)
    return correctedSentence

#Discuss any topic from the dataset with chatbot
discussion_df=pd.read_excel(r'game.xlsx')
discussion_df.columns=["Topic","Content"]
global discussion_documents
discussion_documents={}

for index,row in discussion_df.iterrows():
        if pd.isnull(discussion_df.values).any(1)[index]:
            row["Content"]="I dont know"
            
        if row["Topic"] in discussion_documents:
            discussion_documents[row["Topic"]]= discussion_documents[row["Topic"]] + " " + row["Content"]
        else:
            discussion_documents[row["Topic"]]= row["Content"];

all_topicsOfDiscussion=discussion_documents.keys()

def discussion():
    print("----------------------------------------------------------------------------------------------")
    print("Lets have a discussion. I will say something related to that topic and you will also have to say something related to that topic")
    print("You can not use the same sentence again and again.")
    print("----------------------------------------------------------------------------------------------")
    topic_discussion= random.choice(list(all_topicsOfDiscussion))
    print("Topic of our discussion is :",topic_discussion)
    
    discussion_documents_lines=discussion_documents[topic_discussion].split(".")
    discussion_vocabulary=createVocabulary(discussion_documents[topic_discussion])
    bowOfAllPoints ={}
    for index in range(0,len(discussion_documents_lines)):
        bowOfAllPoints[index] = np.zeros(len(discussion_vocabulary))   # list of same size of vocabulary
        for stem in discussion_documents_lines[index].split():
            try:
                position= discussion_vocabulary.index(stem)
                bowOfAllPoints[index][position] += 1
            except ValueError:
                continue
    allLines=[]
    allFirstLine=[]
    discussion_flag=True
    chatbotPoint=random.choice(discussion_documents_lines)
    allFirstLine.append(chatbotPoint)
    while discussion_flag:
        print("Max:",end=" ")
        for line in discussion_documents_lines:
            if(allFirstLine==None): continue
            for previousline in allLines:
                sim= 1/(1+nltk.jaccard_distance(set(line), set(previousline)))
                if sim>0.6:
                    continue
                else:
                    chatbotPoint=random.choice(discussion_documents_lines)
        print(chatbotPoint)
        print(name,":",end="")
        user_point=input()
        allLines.append(chatbotPoint)
        if user_point.lower()=="i quit":
            discussion_flag=False
            print("Max: What do u wanna do now?")
            continue
        bowOFUserPoint=sentence2bow(user_point,discussion_vocabulary)
        evaluateTheUserPoint(bowOFUserPoint,bowOfAllPoints,user_point,chatbotPoint,allLines)
        allLines.append(user_point)
        
#check the similairty and give the score and also check whether the users point is repeated or not        
def evaluateTheUserPoint(bowOFUserPoint,bowOfAllPoints,user_point,chatbotPoint,allLines):
    max_sim=-1
    for bowOfPresentSentence in bowOfAllPoints:
        score=1-spatial.distance.cosine(bowOFUserPoint,bowOfAllPoints.get(bowOfPresentSentence))
        if score>max_sim:
            max_sim=score

    print("Score:",max_sim)

    if max_sim < 0.1 :
            print("Nice try but next time please make it more relatable. I know you can do it.")
    totalPresent=0
    repeated=False
    for chatbotPoints in allLines:
        totalPresent=0
        for words in user_point.split():
            if words in chatbotPoints.split():
                totalPresent=totalPresent+1
        percentagePresent=totalPresent/len(chatbotPoint.split())
        if percentagePresent>0.6:
            repeated=True
    if repeated: print("Max:Please dont repeat. Try to remember something else. Comeeeeeee Onnnnnn.")


#code to get user question
flag=True
print("Hi there, My name is Max.")
print("I am here to help u in your studies. Here are some things we can do together:")
print("1. We can discuss any topic from your syllabus.")
print("2. If u have any doubts or u wanna revise any question u can ask me.Just let me know that a doubt.")
print("3. If u want to test yourself by any random questions. I can help u with that as well.")
print("4. You can ask me to book your exam.")
print("6. You can share ur problem with me. I will be always there for u to motivate and encourage u.")
print("Well, before we start. What is your name?")
print("P.S- Let me know at the end if u like me or not.")
nameFlag=True
name=""
gkQuestion=False
while(flag):
    print(name,": ",end="")
    user_response = input()
    user_response=user_response.lower()
    if(predict_class(user_response)!='goodbye'):
        if(predict_class(user_response)=='thanks'):
            print("Max: You are welcome..",name)
        elif(nameFlag):
            name=getName(user_response)
            if(name==None): name=user_response.capitalize()
            print("Max: Hi",name,"How can I help u today?")
            nameFlag=False
        elif(predict_class(user_response)=='discussion'):    
            discussion() 
            gkQuestion=False
        elif(predict_class(user_response)=='booking'):
            gkQuestion=False
            testbooking=storeData()
            print(testbooking)
        elif(predict_class(user_response)=='small talk'):
             print("Max: Great! lets have a small talk")
             gkQuestion= False
        elif(predict_class(user_response)=='game'):
             gkQuestion=False
             print("Max: Lets play the game")
             gameCall()
        elif(predict_class(user_response)=='ask'):
            gkQuestion=False
            question=random.choice(list(qa_documents.items()))[0]
            print(question)
            print(name,": ",end="")
            user_answer=input()
            answerSim=checkTheSimilarityOFAnswers(user_answer,question)
            questionSim=checkTheSimilarityOFQuestion(user_answer,question)
            if(questionSim>0.5):
                print("Max: Dont use too much words from question.")
            askedQuestions[question]=answerSim
            print("Score :",answerSim*100)
            if(answerSim<0.4): 
                print("Max: Dont worry. Try harder next time.")
            else:
                print("Max: Good job.")
            print("Answer: ",qa_documents[question])
        elif(predict_class(user_response)=='name'):
            gkQuestion=False
            print("Max:",chatbot_response(user_response),"My name is MAX and your name is",name)
        elif(predict_class(user_response)=='gk' or predict_class(user_response)=='doubt'):
            gkQuestion=True
            print("Max:Ok",name,",I will try my best to help you.Go ahead ask anything.")
        else:
            if(not gkQuestion):
                print("Max: "+chatbot_response(user_response))
            else:
                if(not has_numbers(user_response)):
                    user_response = spellCorrector(user_response)
                print("Max: ",end="")
                response(user_response)
    else:
        flag=False
        showResults(askedQuestions)
        print("Max: Bye",name,"!",chatbot_response(user_response))