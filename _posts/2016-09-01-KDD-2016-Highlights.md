---
layout: post
title:  "Highlights from KDD 2016"
categories: conferences
tags: [data-mining, research, conference]
---

![San Fransisco](/assets/san_fransisco.jpg)
_Photo credit: [David Yu](https://www.flickr.com/photos/davidyuweb/15370470163/)_

This August while interning at [Pandora](https://twitter.com/LifeAtPandora) I had the opportunity to attend the [22nd ACM SIGKDD Conference on Knowledge
Discovery and Data Mining (KDD)](http://www.kdd.org/kdd2016/), held in San Fransisco. My manager
[Oscar Celma](https://twitter.com/ocelma) was cool enough to let me attend during my internship, and my research institute,
[SICS](https://www.sics.se/) was cool enough to cover my conference fee, even though I was not presenting.

Like I did
with my post on [last year's ICDM](/conferences/2015/11/23/ICDM-2015-Highlights.html), I'll be providing summaries for
papers I found interesting from the sessions I attended, and provide links to the full text articles, organized by day
and session. One great thing that KDD did this year was to ask the authors to provide 2 minute Youtube videos describing
their papers, so for most of the linked papers you will find the video as well, providing a brief, accessible explanation.

This will be long post so feel free to skip ahead to the sections that are most interesting to you.

* TOC
{:toc}

## Pre-conference and tutorial days

The Broadening Participation in Data Mining Workshop [(BPDM)](http://www.dataminingshop.com/web/) was held on Friday and
Saturday. This
workshop aims to broaden the participation of minority and underrepresented groups in Data Mining, by providing guidance,
networking and other opportunities. I think it's a great initiative and I hope to see other venues take up something
similar in ML in the vein of [WiML](http://wimlworkshop.org/).

Saturday was tutorials day at KDD, and there were a lot to choose from. I spent most of my time in the [Scalable Learning of
Graphical Models](http://www.francois-petitjean.com/Research/KDD-2016-Tutorial/) and the [IoT Big Data Stream
Mining](https://sites.google.com/site/iotminingtutorial/) tutorials. The deep learning tutorial from Ruslan Salakhutdinov
had to be cancelled (if anyone knows why let me know). The slides and video from his [2014 tutorial at KDD](http://www.cs.toronto.edu/~rsalakhu/kdd.html) are available
however, and I can definitely recommend it as a good introduction to the field.

## Sunday: Workshops and Opening

Sunday was the workshop day and again it made me wish I could split myself to cover many in parallel, with topics like
large-scale sports analytics, learning from time series, deep learning from data mining, and stream mining. In the end,
I chose to spend most of my day in the workshop on ["Mining and Learning from Graphs"](http://www.mlgworkshop.org/2016/), which was closest
to my interests, and probably the best of the day.

### Mining and learning with graphs

#### Keynotes

The reason I think this was the best workshop of the day has a lot to do with the great keynote lineup as well the
quality and variety of the work presented. Lars Backstrom, the director of engineering at Facebook, had the first keynote
where he talked about the challenges in creating a personalized newsfeed for over a billion users. He talked about how
both business decisions and probabilities calculated by many models (trees, deep learning, logistic regression) affect
the scoring of items in the News Feed that end up determining the ranking of the items users see.

He also mentioned
some work I was not previously familiar with, co-authored with Jon Kleinberg, on [discovering strong ties
in a social network](https://dl.acm.org/citation.cfm?id=2531642), like romantic relationships. For this they developed a new measure of tie strength, _dispersion_,
which measures "the extent to which two people's mutual friends are not themselves well-connected". Using this method
they were able to identify the spouse of male users correctly with .667 precision, which is impressive considering they
are only using the graph structure as information. The dispersion metric itself is an interesting concept  and can also be used
for News Feed ranking.

The rest of the keynotes were full of great ideas as well. [Leman Akoglu](https://www.cs.cmu.edu/~lakoglu/index.html), who
recently moved back to CMU after Stony Brook, gave a talk on [detecting anomalous neighborhoods on attributed networks](https://arxiv.org/abs/1601.06711).
[Tamara Kolda](http://www.sandia.gov/~tgkolda/) talked about how to correctly model networks, and presented their [BTER generative graph model](https://arxiv.org/abs/1302.6636) which is able to generate graphs that closely follow properties of
real-world graphs such as degree distribution and the triangle distribution, and a more recent extension of the model to
[bi-partite graphs with community structure](https://arxiv.org/abs/1607.08673).

[Yishou Sun](http://web.cs.ucla.edu/~yzsun/) (now at UCLA) presented a [probabilistic model for event likelihood](http://web.cs.ucla.edu/~yzsun/papers/ijcai16_anomaly.pdf).
The key idea here is
that one can model an event, say a user purchasing an item, as an [ego-network](http://www.analytictech.com/networks/egonet.htm)
(networks where we focus on one node, the "ego" node). The ego node would be the event, linked to heterogeneous entities,
like the item, date, and user. The entities are then embedded into a latent space by using their co-occurrence with other
events, and the embeddings can then be used for tasks like anomaly detection and content-based recommendation.

[Jennifer Neville](https://www.cs.purdue.edu/homes/neville/) presented methods for modelling distributions of networks,
which essentially allows one to [generate network samples](http://www.kdd.org/kdd2016/subtopic/view/sampling-of-attributed-networks-from-hierarchical-generative-models)
of attributed hierarchical networks, which can then be used for inference and evaluation.
Finally, [S.V.N. Vishwanathan](https://users.soe.ucsc.edu/~vishy/) had a disclaimer that his talk was not exactly on the
topic of graphs, but rather how to exploit the computational graph to achieve better parallelism in distributed machine
learning. He presented some recent work on [distributed stochastic variational inference](https://arxiv.org/abs/1605.09499)
that only updates a small part of the model for each data point (compared to classic stochastic VI), to achieve both data and model
parallelism while maintaining high accuracy.

#### Papers

A number of cool ideas were presented through the papers at the workshop:

* Cohen et al.
presented a new algorithm on [distance-based influence in networks](http://www.mlgworkshop.org/2016/paper/MLG2016_paper_35.pdf), where a scalable
influence maximization algorithm was presented which can be used with any decay function.
* Qian et al. presented
a fun idea: [blinking graphs](http://www.mlgworkshop.org/2016/paper/MLG2016_paper_18.pdf). A graph that blinks is a one
where each edge and node exists with a probability equal to its weight. This is then used to provide a proximity measure
between nodes, that turns out to provide outputs that are more intuitive and are shown to be useful in tasks like link
prediction.
* Giselle Zeno used the work presented earlier by J. Neville that allows for the generation of attributed
networks, to create different samples from a network distribution and [systematically study](http://www.mlgworkshop.org/2016/paper/MLG2016_paper_27.pdf) how graph characteristics
affect the performance of collective classification algorithms.
* Rossi et al. presented [Relational Similarity Machines](http://www.mlgworkshop.org/2016/paper/MLG2016_paper_33.pdf),
a model for relational learning that can handle large graphs and is flexible in terms of learning tasks, constraints and
domains.

I would definitely encourage you to take a look at the [workshop website](http://www.mlgworkshop.org/2016/) and check
out some more of the papers. Overall, great work from all the organisers, with a great intro from [Sean Taylor](https://twitter.com/seanjtaylor)
from Facebook, and a diverse and engaging set of keynote speakers. I'll be looking to submit here next year!

## Monday: Day 1 of the main conference

#### Graphs and Rich Data (Best paper award)

I started Day 1 of the conference by attending the Graphs and Rich Data session. The first paper presented was the best
paper award winner, [FRAUDAR: Bounding Graph Fraud in the Face of Camouflage](http://www.kdd.org/kdd2016/subtopic/view/fraudar-bounding-graph-fraud-in-the-face-of-camouflage)
from Christos Faloutsos' lab at CMU. In the paper Hooi et al. describe a method for detecting fraud, in the form of
reviews on Amazon or followers on Twitter, in the presence of camouflage: when fraudulent users have taken over legitimate
user accounts. In the paper they propose a number of metrics to measure the suspiciousness of subsets of nodes in a bipartite
graph (e.g. users and products) and show how to compute them in linear time. They illustrate the effectiveness of the approach
by using a Twitter graph with ~42M users and ~1.5B edges and showing that their algorithm is able to detect a group of
fraudulent users (manually evaluated). I would have loved to see some comparison in terms of accuracy on the real-world
data with other algorithms and a more quantitative evaluation using real-world data, but obtaining that would be hard
without a good ground-truth dataset, and I don't know if any exist for graph-based fraud detection.

#### Large-scale Data Mining

I then moved on to the Large Scale Data Mining session, just in time to catch Daniel Ting deliver a smooth presentation
of his work on [cardinality estimation of unions and intersection with sketches](http://www.kdd.org/kdd2016/subtopic/view/towards-optimal-cardinality-estimation-of-unions-and-intersections-with-ske).
The cardinality of unions and intersections can be used for a number of applications, from calculating the Jaccard
similarity between two sets, to estimating the number of users accessing a particular website grouped by location or time,
and can be used for fundamental problems like estimating the size of a join. Daniel here proposed two new estimators
based on pseudo-likelihood and re-weighted estimators. The re-weighted estimators are perhaps the most interesting as they
can be generalized more easily (the work focuses on the MinCount sketch) and are easier to implement. I particularly like
the main idea behind them: Taking the weighted average of the several estimators after finding  the most uncorrelated ones.
It is a rare thing to see a single author paper nowadays and Daniel hit it out of the park in terms
of quality and rigour with this one.

<!---
Two other great papers from the session were [efficient anomaly detection in streaming graphs](http://www.kdd.org/kdd2016/subtopic/view/fast-memory-efficient-anomaly-detection-in-streaming-heterogeneous-graphs)
from Emaad Manzoor, and the
[XGBoost paper](http://www.kdd.org/kdd2016/subtopic/view/xgboost-a-scalable-tree-boosting-system) from Tianqi Chen. Emaad presented StreamSpot, an anomaly detection approach for streaming heterogeneous
graphs. He uses a string representation (shingles) for local substructure of graphs, and then uses a variation of SimHash named
StreamHash to compute similarities between the shingles. The algorithm is then initialized with benign clusters and the
anomalies then are detected for each cluster based on their deviation from the cluster's graph and medoid. My impression
is that the initialization process requiring a benign dataset limits the applicability of the algorithm somewhat, since
one can never be sure a dataset does not contain any anomalies, unless it is completely hand-labeled. Still the idea is
novel and I liked the translation of graphs to shingles along with the StreamHash algorithm.
--->

In the same session Tianqi Chen presented [XGBoost]((http://www.kdd.org/kdd2016/subtopic/view/xgboost-a-scalable-tree-boosting-system)). 
I assume [XGBoost](https://xgboost.readthedocs.io) needs no introduction to most, it's a gradient boosted tree algorithm
that has become wildly popular and has been used in the winning solution for 17 out of 29 Kaggle challenges during 2015.
Part of the appeal of XGBoost lies in its scalable nature and Tianqi has gone to great lengths
to ensure the algorithm is fast, easy to use and will run from anywhere
(C++, Python, R) and on anything  (local and distributed). JVM-based solutions were also added recently, so it is possible
now to XGBoost on top of [Apache Flink](http://dmlc.ml/2016/03/14/xgboost4j-portable-distributed-xgboost-in-spark-flink-and-dataflow.html)
or Spark for example. I hope to find the time this year to work on the Flink integration so that it becomes a great platform
to run on and boosts our efforts (pun intended) for [ML on Flink](https://ci.apache.org/projects/flink/flink-docs-master/dev/libs/ml/index.html).

#### Streams and temporal evolution I (Best student paper award)

Lorenzo De Stefani from Brown presented [TRIÈST](http://www.kdd.org/kdd2016/subtopic/view/triest-counting-local-and-global-triangles-in-fully-dynamic-streams-with-fi),
a new algorithm for counting local and global triangles in streaming graphs,
that supports additions and deletions of edges, with a fixed memory budget.
Counting triangles is a classic problem in network theory, as it can help with many tasks like spam detection, link
prediction etc. In many real world graphs, like a social network, edges are constantly being added and removed, so
maintaining an accurate count of the triangles in real-time is a challenging problem, especially in graphs with millions
of nodes and billions of edges.

What De Stefani et al. have done
here is present a one-pass algorithm based on reservoir sampling that provides unbiased estimates of the local and global
triangle counts with very little variance, that only requires the user to specify the amount of memory they want to use (an easy parameter to set).
Compared to previous approaches, TRIÈST does not require the user to set an edge sampling probability (a parameter that is
very hard to set without prior knowledge about the stream), and provides full
utilization of the available memory early on (vs. the end of the stream).
I find the use of reservoir sampling a great "oh why didn't I think of that" idea here, and the value of the paper comes
from the rigorous analysis of the algorithm, and the extensive experimentation the authors have performed.
A very worthy recipient of the best student paper award.

#### Streams and temporal evolution II (Theo's coolest idea of KDD award)

Perhaps the most novel idea I saw at KDD came from the paper on [Continuous Experience-aware Language modelling](http://www.kdd.org/kdd2016/subtopic/view/continuous-experience-aware-language-model)
by Mukherjee et al. from [MPI](https://www.mpi-inf.mpg.de/home/). The idea here is to try to model the experience of the user in reviewing items from a particular domain,
based on the evolution of their language model. Think of a beer reviewing site. Your first few reviews might contain sentences
like _"I like this beer"_ or _"Great taste!"_. But as you gain more experience in tasting beer, the way you describe it
becomes more nuanced; you might write something like _"Fascinating malt and hoppiness, the aftertaste left something to be desired
however"_. So as you evolve as a  beer drinker, so does the language you use to describe it.

Previous work in the field has
tried to model this evolution of experience on a discrete scale; the user's experience remains either static or suddenly
jumps a level. In this work the authors have used a model used in financial analysis called [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
to instead model the evolution of the user's experience as a continuous-time stochastic process. The user's language
model is also continuous, using a dynamic variant of LDA that employs variational methods like Kalman filtering for inference.
Using this model they are able to more accurately recommend items to users, (albeit using RMSE as a metric which
[was shown to be problematic](https://www.researchgate.net/profile/Paolo_Cremonesi/publication/221141030_Performance_of_recommender_algorithms_on_top-N_recommendation_tasks/links/55ef4ac808ae0af8ee1b1bd0.pdf))
and do some explorative analysis, like show the evolution of term usage with experience, or the top words used for experienced
and inexperienced users. Overall I really liked this idea of tracking the language model of users over time, and I
believe that continuous-time models can have beneficial effects in many other domains.

## Tuesday: Day 2 of the main conference

#### Deep learning and embedding

The paper describing the system that [Inbox by Gmail](https://www.google.com/inbox/) uses for its [Smart Reply automated 
response system](http://www.kdd.org/kdd2016/subtopic/view/smart-reply-automated-response-suggestion-for-email)
was presented on Tuesday and it drew a lot of attention as expected. In case you are not familiar with the system, it
provides users of the Inbox app with short, automated replies. So the app writes the
emails for you instead of you having to type them out (yay machine learning!). This is obviously a highly challenging
tasks for many reasons. How can you tell if an email is a good candidate for a concise response? How does one generate
a response that is relevant to the incoming email? How does one provide enough variance in the responses generated?
And as is always the case at Google, how does one do this at scale? 

Kannan et al. describe a system that uses a feed-forward
neural net to determine if an email is a good candidate to show automated responses for, an LSTM network for the actual
response text generation (sequence-to-sequence learning), a semi-supervised graph learning system to generate the set of
responses, and a simple strategy to ensure that the responses shown to the user are diverse in terms of intent. Although
the paper does not delve very deeply into each topic as they have to cover a complicated end-to-end learning system, it's
still a great read as it provides insights into the scalability issues with deploying such models to millions of users,
as well as the challenge of optimizing for multiple objectives (accuracy, diversity, scalability) in a complex system.

#### Recommender Systems

In this session chaired by [Xavier Amatriain](https://twitter.com/xamat/), [Konstantina Christakopoulou](http://www-users.cs.umn.edu/~christa/) presented her paper
on [conversational recommender systems](http://www.kdd.org/kdd2016/subtopic/view/towards-conversational-recommender-systems).
The scenario here is common: You are at a new city, and would like to go out for dinner. If you had a local friend,
you'd have a small conversation: "Do you like Indian? What about Chinese? What's your price-range?" and based on your
responses you knowledgeable friend would recommend a restaurant that they think you'd like. The challenges in creating
an automated system that does this are many: How does one find which dimensions are important (cuisine, price)?
Which questions should the system pose in order to arrive to a good recommendation as soon as possible?

Konstantina
addresses this problem as an online learning problem, where the system learns the preferences of the user online,
as well as the questions that allow it to provide good recommendations quickly. This is done by utilizing a bandit-based
approach that adapts the latent recommendation space to the user according to their interactions, and a number of
question selection strategies are tested, where is it shown that using a bandit-like approach to balance exploration
and exploitation in the latent question space is highly beneficial. I'm a fan of this work because I think it directly addresses cold-start
problems in recommender systems with an intuitive and human-centered approach, which includes knowledge we already have
about users and items through classic CF systems, with online learning and incorporating context.

#### Turing Lecture: Whitfield Diffie

Since this post is already too long I will not be covering the keynotes, however I could not skip mentioning
Whitfield Diffie's Turing lecture, which was one of the highlights of the conference. Whitfield took us on journey through the
history of cryptography, starting with the [Ceasar Cipher](https://en.wikipedia.org/wiki/Caesar_cipher) all the way to
[Homomorphic encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption), with many interesting tidbits and
historical anecdotes along the way.

I particularly liked his story on one of the things that motivated him to find a solution for the public key cryptography
problem that he is most famous for. Diffie explained that one of his friends had told him that at the NSA the phone lines
are secure, so Diffi thought that without having an encryption key negotiated before-hand you could pick up a phone and dial and your
communication would be safe from eavesdropping. Diffie assumed that hey had somehow solved the problem of key distribution,
which motivated him to work even harder on the problem. The reality was that NSA was simply using shielded private lines
for their communication, but in Diffie's own words, ___"Misunderstanding is the seed of invention"___.

Another problem was presented by Diffie's mentor [John McCarthy](https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist))
at a conference in Bordeaux where he talked about "buying and selling through home terminals", what we call e-commerce today.
This problem led Diffie to think about digital signatures and proof of correctness, and the key idea of having a problem which you cannot
solve, but you can tell whether a solution provided is correct, which eventually led to public key cryptography.
I cannot provide a good enough summary of the talk here, but I would wholeheartedly recommend watching the whole thing, as
[it's up on Youtube](https://www.youtube.com/watch?v=CIZh0CHXGC4) and most definitely worth your time.

## Wednesday: Last day of the conference

#### Supervised learning

The highlight of the supervised learning session was the work from Marco Tulio Ribeiro et al. on [explaining the predictions of
any classifier](http://www.kdd.org/kdd2016/subtopic/view/why-should-i-trust-you-explaining-the-predictions-of-any-classifier).
The problem they are trying to solve is interpretability: Despite the wide adoption of machine learning, many of the more
complicated models, such as deep learning or random forests, are used as black boxes, explaining why they gave us a
particular answer is very hard. This makes it difficult to trust the system, and deploy it in a setting where it would aid
critical decision-making, like whether or not to administer a specific treatment to a patient.

The proposed system, [LIME](https://github.com/marcotcr/lime), which stands for Local Interpretable Model-agnostic Explanations, can explain the outputs of
any classifier. The way they achieve that is by fitting a simple, interpretable model like a linear regression, on generated
samples, weighted by their distance to the prediction point. What this essentially does is to approximate the complex
decision boundary locally using an interpretable model, from which we can then explain why a decision was made. For text
this could be the words that were present in a document that lead us to classify it as spam or not, and in images it could
be superpixels that caused the classification of the image as containing as dog or cat. The system introduces a lot of
overhead of course, the authors report 10 minutes runtime to explain one output from InceptionNet on a laptop, but there is a lot of room
for improvement there. Interpretability is one of the main challenges for ML in the coming years and it's always welcome
to see new exciting work on the subject.

#### Optimization

One of the best papers of the conference was presented in one of the final sessions by Steffen Rendle, of factorization
machines fame, who is now at Google. He and his colleagues provide a solution for a problem of scale: How to train a
generalized linear model in a few hours for a trillion examples. For this they proposed [Scalable Coordinate Descent (SCD)](http://www.kdd.org/kdd2016/subtopic/view/robust-large-scale-machine-learning-in-the-cloud),
whose convergence behavior does not change regardless of the how much it is scaled out or the computing environment.
They also described a distributed learning system designed for the cloud which takes into consideration the challenges
present in a cloud environment, like shared machines (VMs) that are pre-emptible (i.e. you could be kicked out after a
grace period), machine failures etc.

The problem with [coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent) is that it's a highly sequential
algorithm, and not a lot of work has to be done at each step, which make parallelizing or distributing it challenging.
The key idea for the SCD algorithm is to make use of the structure and sparsity
present in the data. The data are partitioned in "pure" blocks where each block has at most one non-zero entry, and
updates are performed per block. The enforced independence in features is what enables the parallelism for the algorithm.
On the systems side they use a number of tricks to deal with having short barriers. Syncing the workers is challenging
in the presence of stragglers (machines that are slower) which they overcome by using dynamic load-balancing, caching,
and pre-fetching. Using this system and algorithm they are able to achieve near-linear scale out and speed up, going
from 20 billion examples to 1 trillion.

## Closing thoughts

Overall the conference was well organized and a pleasure to attend. The venue was great, even though many of the sessions
had to be done in a different hotel across the street. The choice of having some of the keynotes over lunch however was criticised
by most attendees, as it was almost impossible to hear the speakers, and I'm sure it was not a good experience for them
either. The conference had a very heavy company presence as well, which I actually found welcome, as I had the opportunity
to talk to people from many interesting companies who are doing great research work like Microsoft, Facebook, Amazon etc.


If I have one gripe with conference is the insistence _"per KDD tradition"_ on not performing double-blinded or open reviews, even though
[the research community](https://hub.wiley.com/community/exchanges/discover/blog/2016/06/27/what-are-the-current-attitudes-toward-peer-review-publishing-research-consortium-survey-results)
is moving towards that ([original paper]((https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4191873/))),
so kudos to ICLR for open and ICDM for triple-blind reviews.

In closing, KDD was a great conference and I'm glad I was given the opportunity to attend. I met a bunch of great new people and
reconnected with old friends, had interesting discussions with many companies, and the research presented filled me with
new ideas to take home and expand.

Looking forward to next year!


