# MLNewsBot

This is an intelligent Twitter Bot that tracks ML related hashtags in real time and intelligently detects tweets that are not spam using NLP. It then posts reposts these tweets so its followers can have a useful source of ML/AI news and tips.
The NLP has been built into a REST API and deployed on Heroku for free public web hosting. The bot sends requests to the URL to classify tweets.
The bot has been Dockerized and deployed on Azure. Next steps include: 
1. Improving the NLP by performing an ML analysis with other models, such as fbprophet and LSTMs
2. Using ONNX to ensure model interoperability and optimize Azure hardware usage
3. Adding additional microservices
4. Creating an AKS (Kubernetes cluster) to better manage containers and scale the bot.  
The current event flow is shown below:
```mermaid
sequenceDiagram
Http Request ->> Azure Hosted Container FQDN: set tweet interval time
Http Request ->> Azure Hosted Container FQDN: start bot
Azure Hosted Container FQDN ->> MLNewsBot (Tweepy Stream): start streaming
MLNewsBot (Tweepy Stream) ->> Twitter: open stream
Twitter ->> MLNewsBot (Tweepy Stream): tweet data
MLNewsBot (Tweepy Stream)-->> Heroku Custom RestAPI: get tweet classification (spam/not spam)
Heroku Custom RestAPI-->> MLNewsBot (Tweepy Stream): return classification
MLNewsBot (Tweepy Stream) -->> MLNewsBot (Tweepy Stream):check if classification is not spam
MLNewsBot (Tweepy Stream) -->> MLNewsBot (Tweepy Stream):check if time interval is satisfied
MLNewsBot (Tweepy Stream) -->> Twitter Handle: if both conditions satisfied, retweet tweet
```
