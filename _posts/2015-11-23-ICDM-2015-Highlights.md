---
layout: post
title:  "Highlights from ICDM 2015"
categories: conferences
tags: data- mining research
---

This past week I had the opportunity to attend the [15th IEEE International Conference on
Data Mining](http://icdm2015.stonybrook.edu/), held in Atlantic City, NJ, November 14-17, 2015.
This was the first scientific conference I attended and we had a chance to present our
work on scalable graph similarity calculation. In this post I will try to point out some
of the more interesting work from the conference (based on some of the sessions I attended)
and summarize the keynotes. I've included links to the full-text papers whenever I could
find them.

## Highlights from the sessions I attended:

## Day 1

# Applications 1

The first session I attended was named "Applications 1" and it included
a number of applications (surprise!) on a diverse set of domains. The session started
with some very solid work on ["Modeling Adoption and Usage of Competing Products"](http://arxiv.org/abs/1406.0516),
where the authors create a model that can provide insight into the factors that drive
product adoption and frequency of use, which they evaluate at a large scale by looking
into the use of URL shorteners on Twitter.
In *"Mining Indecisiveness in Customer Behaviors"* the authors investigated how they could
reduce indecisiveness in users interacting with an online retail platform, by making use
of information about competing products. The end goal is to increase conversion of course,
but it would be interesting to see how such a system could be implemented in a way that
is fair to all retailers/brands.

Two short papers I should point out were ["Personalized Grade Prediction: A Data Mining
Approach"](http://medianetlab.ee.ucla.edu/papers/Yannick_ICDM.pdf) and
["Sparse Hierarchical Tucker and its Application to Healthcare"](http://www.cc.gatech.edu/~iperros3/publications/icdm15.pdf).
The first
paper deals with personalized early grade prediction for students using only assignment/homework data,
that could allow course instructors to identify students who might have
trouble in a course early on, most importantly using only their data from the specific
course, thereby avoiding any potential privacy pitfalls. In the second work, a new tensor
factorization method is proposed, that is 18x more accurate and 7.5x faster than the current
state-of-the-art. While the application presented here is limited to healthcare, I hope
that it can prove a starting point for a more generalized approach, as tensor factorization
problems can surface in wide variety of domains so solving their scalability problems
could have an effect on a wide range of fields.

# Mining Social Networks 1

The next session I attended was "Mining Social Networks 1", where the best student paper,
["From Micro to Macro: Uncovering and Predicting Information Cascading Process with
Behavioral Dynamics"](http://arxiv.org/abs/1505.07193) was presented among others.
Cascade prediction has applications
in areas like viral marketing and epidemic prevention, so it's a problem of great interest
in the industry as well as society. The work presented here utilized a data-driven approach
to create a "Networked Weibull Regression" model, and use it for predicting cascades
as they occur, going from micro behavioral dynamics modelling which are aggregated to predict
the macro cascading processes.

They evaluate their method on a dataset from Weibo, one of the largest Twitter-style
services in China, and show that their method handily beats the current state of the art.
It's a well written work that deserves the praise it got, however I would definitely be interested
in seeing it applied and evaluated on a different publicly available dataset, (although they are
hard to come by in this domain) and an extension of the method that predicts the cascades as they
happen in real-time (shameless plug: Use [Apache Flink](flink.apache.org) for your real-time processing needs!).

# Big Data 2

The last session I attended on Sunday was "Big Data 2". The two regular papers from that
session were perhaps application specific but nonetheless provided some valuable insights.
The first, "Accelerating Exact Similarity Search on CPU-GPU Systems" dealt with the exact
kNN problem, and how it can be efficiently accelerated on GPU-equipped systems. Although
approximate kNN methods like LSH seem to be the standard at the industry currently, the
authors mentioned that the techniques presented could be used in that context as well,
so this is something to look forward to definitely. The second regular paper ["Online Model
Evaluation in a Large-Scale Computational Advertising Platform"](http://arxiv.org/abs/1508.07678)
provided a rare look into how a large advertising platform like Turn evaluates its bid prediction models online,
something that a previous related paper from Google,
["Ad Click Prediction: a View from the Trenches"](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf),
was missing.

## Day 2

# Network Mining 1

An interesting idea presented in "Network Mining 1" was ["Absorbing random-walk centrality"](http://arxiv.org/abs/1509.02533),
where the authors presented a way to identify _teams_  of central nodes in a graph. An application
for this measure could be for example: given a subgraph of Twitter that we know contains
a number of accounts about politics, find hte important nodes that represent a diverse set
of political views. The authors show that this is an NP-hard problem, and the greedy algorithm
presented has a complexity of O(n^3), where n is the number of nodes, which makes it
inapplicable for large graphs. Personalized PageRank could be used as heuristic however which
is more computationally efficient.

## Day 3

# Graph Mining

We presented out work, ["Knowing an Object by the Company It Keeps: A Domain-Agnostic Scheme for Similarity Discovery"](/assets/concepts-icdm.pdf),
in the "Graph Mining" session. Our main contribution is a method that allows us to
transform a _correlation graph_ to a _similarity graph_, where connected items should be _exchangable_ in some sense.

As an example, think of a correlation graph where we have words as nodes and edges between words are created by taking the conditional probability of a word appearing
within _n_ words of another one. This can be easily extracted from a text corpus and pairs like (_Rooney, goal_)
could have a high correlation score. What we want to do with our algorithm is to discover _similarities_
between items that go beyond simple correlation, and show characteristics such as exchangability.
For example a pair (_Rooney_, _Ronaldo_) could be a good pair in this sense, as you could replace
Rooney with Ronaldo in a sentence and it should still make sense. The approach we presented is domain
agnostic, and as such is not limited to just text; we applied our algorithm on graphs of music artists and [codons](https://en.wikipedia.org/wiki/Genetic_code)
as well. I will soon write up a more extensive summary of our work, including code and examples.
For now enjoy this [nice visualization](/assets/concepts-visualization.pdf)
of word relations and clusters that can be created using our method.
*Note:* better to download and view in a PDF viewer which has *lots* of zoom.

Some impressive work for me from that session was ["Efficient Graphlet Counting for Large Networks"](http://arxiv.org/abs/1506.04322).
[Graphlets](https://en.wikipedia.org/wiki/Graphlets) are small, connected, induced (i.e. the edges
in the graphlet correspond to those in the large graph) subgraphs of a large network, and can be used
for things like graph comparison and classification. The method presented here uses already proven
combinatorial arguments to reduce the number of graphlets one has to count for every edge, and
obtains the remaining counts in constant time. In a large study of over 300 networks the algorithm
is shown to be on average 460 times faster that the current state-of-the-art, allowing the largest
graphlet computations to date. I am always happy when I see established results used in a clever
way to solve new problems, especially when the results are so impressive.


## Keynotes

# Robert F. Engle

ICDM featured 3 keynotes this year. The first one was given by Robert F. Engle, winner of the
Nobel Memorial Prize in Economic Sciences in 2003. He presented a summary of some of his seminal
work on [ARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity),
and presented some more recent work on financial volatility measurement through the [V-lab](http://vlab.stern.nyu.edu/)
project. This keynote was quite math-havey as a result and I think many people in the audience did
not find it that interesting/relevant to their work, estimated from the proportion of people with looking at their
laptops around me.

# Michael I. Jordan

The second keynote, and the most interesting for me, was given by M.I. Jordan, with the title "On
Computational Thinking, Inferential Thinking and 'Big Data'", a talk he has delivered in a couple
of other venues before, so (some of) the [slides are available](http://www.stat.harvard.edu/NRC2014/MichaelJordan.pdf).
His keynote revolved around some of what he identified as central demands for learning and inference
and the tradeoffs between them; namely error bounds ("inferential quality"),
scaling/runtime/communication, and privacy. He identified the problem of lack of an interface
between statistical theory and computational theory which currently have an "oil/water"
relationship, where in one more data points are great as they reduce uncertainty, and can be a cause
of problems in the other as we usually measure complexity in the order of data points. The approach
suggested is to to "treat computation, communication, and privacy as constraints on statistical
risk".

In terms of privacy he mentioned how our inference problem basically has 3 components, the
population P, which we try to approximate with our sample S, which we then modify
according to our privacy concerns to get our final dataset Q, which we can query.
In dealing with privacy issues he mentioned [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy)
as a good way to quantify the privacy loss for a query. This should allow us, given a some privacy
concerns, to estimate the amount of data we need to achieve the same level of risk in our queries.

For the tradeoff between inferential quality and communication, a common tradeoff in distributed
learning settings, he proposed the use of a channel with certain communication constraints, as a
way to impose bitrate constraints. The proposed solution involves minimax risk with B-bounded
communication, which allows for optimal estimation under a communication constraint (see
[here](http://www.cs.berkeley.edu/~yuczhang/files/nips13_communication.pdf)
for the NIPS paper on the subject).

The last part of his talk was new (i.e. is not the slides linked) and concerned the tradeoff between
inference quality and computation resources. This part focused on efficient distributed bootstrap
processes, with the thesis being that such processes can be used to generate multiple realizations
of samples from the population that allow for the efficient estimation of parameters. The problem
with a frequentist approach in this case is that the communication cost of each resampling can be
prohibitively high for large datasets, e.g. ~623GB for a 1TB dataset
(see [here](http://www.stat.washington.edu/courses/stat527/s13/readings/EfronTibshirani_JASA_1997.pdf) why).
The proposed solution here is the "[Bag of Little Bootstraps](http://web.cs.ucla.edu/~ameet/blb_icml2012_final.pdf)",
in which one bootstraps many small subsets of the data and performs multiple computations on these
small samples. The results from these computations are then averaged to obtain an estimate of the
parameters of the population.
This means that in a distributed setting we would use only small subsets of the data to perform
our computation; in the 1TB example above, the resample size could for example be 4GB instead of
the 632GB required by the bootstrap.
Another interesting point made was that obtaining a confidence interval on a parameter instead of
a point estimate like is usually done now, can not only be more useful, but could be done more
efficiently as well.

In closing Jordan identified there are many remaining conceptual and mathematical challenges in the
problem of 'Big Data' and facing these will require a "rapprochement between computer science and
statistics" which would reshape both disciplines and might take decades to complete.

# Lada Adamic

Unfortunately I had to skip Lada Adamic's keynote, so I would really appreciate if someone has a
summary that I can add here.


## Venue/Organisation

The conference organization was mostly smooth and organizers and volunteers deserve a lot of credit for
the way that everything worked out. Sessions generally began and ended on time, the workshops and
tutorials were well organized and useful, and I particularly enjoyed the PhD forum.
One thing that I found unusual was the fact that even though the proceedings were handed out in
digital form (kudos for that) attendees had to choose between the conference or the workshop
proceedings. My guess is this was for licensing cost issues, but it would have been nice to have
access to both.

The conference this year took place at Bally's casino/hotel in Atlantic City.
It was hard to avoid the grumbling from many of the participants for the choice of venue, especially
when one puts it next to last year's venue in [Shenzen](/assets/shenzen.jpg) or next year in
[Barcelona](/assets/barcelona.jpg).

Truth be told, the venue was underwhelming, but I guess it was mostly the choice of Atlantic City
that had people irked; there was very little to do and see in the city unless you wanted to gamble.
Still, I was fortunate to meet a lot of cool people at the conference, so I'm looking forward to
attending next year's edition in Barcelona!

There was a lot of other great work at the conference as well, but these were the presentations
I found most memorable.
So that's all for now. If I've made a terrible mistake when describing your work, shoot me an [email](mailto:tvas@sics.se)
and I'll fix it ASAP.

