import logging
import os

from telegram import InlineKeyboardButton, InlineKeyboardMarkup #, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler


# -------------------- script for A.I. -----------------------#
import numpy
import pandas
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer

DATASET = os.environ.get('DATASET')
ps = SnowballStemmer('english')

def preprocess(text):
            # Stem and remove stopwords
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            return ' '.join(text)
  
if DATASET=="testdata":
    dataset = pandas.read_csv('Wizard-of-Oz-dataset - Test Questions.csv', encoding='utf-8-sig')
elif DATASET=="oct18interviews":
    dataset1 = pandas.read_csv('interview1.csv', encoding='ISO-8859-1')
    dataset2 = pandas.read_csv('interview2.csv', encoding='ISO-8859-1')
    dataset3 = pandas.read_csv('interview3.csv', encoding='ISO-8859-1')
    
    dataset = pandas.concat([dataset1, dataset2, dataset3])
    #Reset index otherwise during the loop below we select multiple rows (pandas.concat results in repeated indices)
    dataset = dataset.reset_index(drop=True)
elif DATASET=="mdc": ##For 'margarita dialogue corpus'
    dataset = pandas.read_csv('MargaritaCorpusKB-bert.csv', encoding='utf-8')
    dataset = dataset.rename(columns={"title": "Q", "paragraphs": "A"})
    

querycorpus = []
for i in range(0, len(dataset)):
    query = re.sub('[^a-zA-Z]', ' ', dataset['Q'][i])
    query = query.lower()
    query = query.split()
    query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
    query = ' '.join(query)
    querycorpus.append(query)      

# Creating the Bag of Words model with TFIDF and calc cosine_similarity
vectorizer = CountVectorizer(decode_error="replace")
vec_train = vectorizer.fit_transform(querycorpus) #this is needed to get the attribute vocabulary_
training_vocabulary = vectorizer.vocabulary_
transformer = TfidfTransformer()
trainingvoc_vectorizer = CountVectorizer(decode_error="replace", vocabulary=training_vocabulary)
tfidf_querycorpus = TfidfVectorizer().fit_transform(querycorpus)



def toia_answer(newquery, k=5):
    
    # if newquery=="👎":
    #     return "[Bad feedback recorded]"
    # if newquery=="👍":
    #     return "[Good feedback recorded]"
    
    tfidf_newquery = transformer.fit_transform(trainingvoc_vectorizer.fit_transform(numpy.array([preprocess(newquery)]))) 
    cosine_similarities = cosine_similarity(tfidf_newquery, tfidf_querycorpus)
    related_docs_indices = (-cosine_similarities[0]).argsort()
    sorted_freq = cosine_similarities[0][related_docs_indices]
    
    #note for this distance the problem we had befor with inf, we have now with 0. Again we decide
    #to make the prediction a bit random. This could be adjusted to remove any 0 distance and
    #pick the only ones left if any, and if none predict 1.
    
    if sum(sorted_freq)==0:
        return "Not understood"
    
    elif sorted_freq[k-1]!=sorted_freq[k] or sorted_freq[k-1]==sorted_freq[k]==0:
        selected = related_docs_indices[:k]
       
        return dataset.iloc[selected[0]]['A']
#        return dataset.iloc[selected[0]]['A'], dataset.iloc[selected,:(k-1)]   
#        print("\n Cosine Similarities:", sorted_freq[:k], "\n")
    else:
        indeces = numpy.where(numpy.roll(sorted_freq,1)!=sorted_freq)
        selected = related_docs_indices[:indeces[0][indeces[0]>=k][0]]
    
        return dataset.iloc[selected[0]]['A']
#        return dataset.iloc[selected[0]]['A'], dataset.iloc[selected,:(k-1)]
#        print("\n Cosine Similarities:", sorted_freq[:k], "\n")

#-------------------------------------------#

# custom_keyboard = [["👍", "👎"]]
# reply_markup = ReplyKeyboardMarkup(custom_keyboard)
custom_keyboard = [[InlineKeyboardButton("👍", callback_data='1'), InlineKeyboardButton("👎", callback_data='0')]]
reply_markup = InlineKeyboardMarkup(custom_keyboard)

def start(update, context):
    update.message.reply_text("Hello! This is an experimental environment built by Alberto M. Chierici, researhcer at New York Univeristy Abu Dhabi. We are researching on modeling dialogue and building applications where you can chat with previously recorded human avatars. You'll be able to chat with our first avatar. Please go ahead and get to know the avatar as you would do when meeting a person for the first time. To help us improving the system, when you think the answer you receive is appropriate, please press the 👍 button below the message. If the answer doesn't make sense, give a negative rating by using the 👎 button." )
    
def toia_bot(update, context):
    update.message.reply_text(toia_answer(message.text.encode('utf-8').decode()), reply_markup=reply_markup)

def button(update, context):
    query = update.callback_query
    query.edit_message_text(text="[thanks, rating recorded: {}]".format(query.data))

def help(update, context):
    update.message.reply_text("Use /start to test this bot.")

def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


if __name__ == "__main__":
    # Set these variable to the appropriate values
    TOKEN = os.environ.get('TOKEN')
    NAME = os.environ.get('NAME')

    # Port is given by Heroku
    PORT = os.environ.get('PORT')

    # Enable logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up the Updater
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    # Add handlers
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text, toia_bot))
    dp.add_handler(CallbackQueryHandler(button))
    dp.add_handler(CommandHandler('help', help))
    dp.add_error_handler(error)

    # Start the webhook
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook("https://{}.herokuapp.com/{}".format(NAME, TOKEN))
    updater.idle()
