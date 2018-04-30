---
layout: post
title:  "A Brief History of Information Theory"
categories: articles
tags: [information theory, research, history]
use_math: true
---


> The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point. Frequently the messages have meaning.
> 
>  -Claude Shannon (1948)

As part of a course I'm taking we were tasked with writing the history
of a scientific discovery we find important. I knew immediately what
I wanted to write about: Claude Shannon and the development of information
theory. It's a story that involves some of the greatest minds in
science, the World War, and the genius of one person.

So with the opportunity of Shannon's birthday (April 30th) I decided to post
the essay here. Most of the material was pulled from James Gleick's
excellent: "The Information: A History, A Theory, A Flood" which I recommend
to anyone even remotely connected to computer science.

## Introduction

In the same year that Bell Labs announced the invention of an electronic semiconductor that could do everything a vacuum tube could more efficiently, Claude Shannon published in The Bell System Technical Journal his seminal work, "A Mathematical Theory of Communication". The information age had just begun.

It's hard to summarize the importance of Shannon's work, as it laid the groundwork for what we now call information theory, with implications for practically every field in science today, from engineering, to biology, medicine, and the social sciences. It underlies everything about "the quantification, storage, and communication of information". In it, Shannon proposed a unit for measuring information, which he termed "binary digits" or _bits_ for short. He derived a formula for channel capacity that determines the absolute speed limit of any communication channel. From that limit he showed that it is possible to devise error correcting codes in a noisy channel that will overcome any level of noise. The contributions now underpin the work in compression and coding for error correction used in everything from mobile communication and media compression, to quantum computation.

Apart from its scientific importance, the story of the development of information theory is fascinating in itself, as it involves some of the greatest thinkers in computer science, and provides a look into the birth of "great ideas".

## Earlier Attempts

To understand how Shannon arrived at his theory, one has to look at the scientific context and earlier attempts to quantify information. In the 1830s, the advent of the telegraph led to the creation of coding schemes, like Morse code, that were used to communicate messages over large distances using electrical signals. In those early attempts, before any formal notion of coding theory was developed, Morse code demonstrated the principles of lossless compression, by using shorter messages for more common letters (one dot for "e") than for less common ones ("j" is one dot followed by three dashes).

The earliest attempts to quantify information content was developed in the 1920s by two Swedes working at Bell Labs: Job B. Johnson and Henry Nyquist, work that was later extended by Ralph Hartley. Job B. Johnson observed that the thermal noise in circuits followed the principles of Brownian motion, i.e. a stochastic process, which Nyquist later expanded upon. In Nyquist's 1924 paper, "Certain Factors Affecting Telegraph Speed" he calculates a formula for the "speed of transmission of intelligence", that connects the speed of "intelligence" transmission to the bandwidth of the channel. It was Hartley who in his 1928 paper "Transmission of information" first used the word "information" to describe the "stuff" of communication and to provide a stricter definition for what had until that point a vague term. Information could be words, sound or anything else. A transmission was defined as a sequence of n symbols from an alphabet with S symbols. In this sense information was a quantity, that determined the ability of the receiver of a transmission to determine if a particular sequence was intended by the sender, regardless of the content of the message.

The amount of information is proportional to the the length of the sequence, but depends on the number of possible symbols as well: a symbol from a two-symbol alphabet (0-1) caries less information than a letter of the English alphabet, which caries less information from a Chinese ideogram. The amount of information was defined by Hartley as:

$$
H = n\log{S}
$$

where n is the length of the message and S the size of the alphabet. In a binary system S is two. The relationship between information and alphabet size if logarithmic: a doubling of the information requires quadrupling the alphabet size.

The assumption made by Hartley however was that each symbol had equal probability, and contained no notion about the communication of messages of unequal probability.

## Wartime

The development of information theory by Shannon was a result of many years of previous research and interactions with some of the brightest minds of his era. The foundations were set in the period of the Great War, were Shannon met Alan Turing as part of their work on cryptanalysis.
Turing had been working on deciphering the messages produced by the German "Enigma" machines. These were machines that the Germans used to communicate encrypted messages. The basis of (symmetric) cryptography is the substitution of symbols in the alphabet with different ones, according to a _key_ that sender and receiver share, and must remain secret from adversaries.

At its essence as Shannon noted "a secrecy system is almost identical with a noisy communication system". The job of the cryptanalyst is to take the seemingly random stream of symbols that is the result of encryption, and try to detect _patterns_ that correspond to the original language.

![A general secrecy system [Shannon49b]](/assets/shannon-crypto.jpg)

_Schematic of a general secrecy system [Shannon49b]_


In Shannon's view, patterns equaled to redundancy: If we are fairly certain in the appearance of a symbol after another, the second symbol can be considered redundant. For example, after the letter _t_ the most likely letter to appear is _h_, making the information content of _h_ after a _t_ lower than all the other letters in the alphabet.
The process of deciphering a message then became a practice in pattern matching and probability: As long as the cipher maintained some notion of patterns that exemplified statistical regularity, they could be cracked. Shannon completed his report on "A Mathematical Theory of Cryptography" in 1945, but it would not be declassified until 1949. This work established the scientific principles of cryptography. He showed that a perfect cipher must produce keys that are truly random, each key must be used only once, be as large as the original message, and never re-used. The term "information theory" appears for the first time in this text.

Turing had been working in a similar vein at Bletchley Park, and had defined his own measure of information, the ban (now also called the hartley), which measured information on base 10 logarithms (instead of base two, as does the bit). During 1943 the two men met daily at Bell Labs where Turing was visiting to work on the X system, used to encrypt the communications between Franklin D. Roosevelt and Winston Churchill.
While they were not able to talk about their cryptanalysis work, they exchanged ideas on "thinking machines" and the "halting problem" that Turing had resolved before the war.
The basis of both men's work in cryptanalysis had been the statistical nature of communication, in its patterns, the resulting redundancy, and how those could be exploited to decipher messages. As Shannon himself noted, communication theory and cryptography were "so close together that you couldn't separate them" and promised to develop these results "in a forthcoming memorandum on the transmission of information".

## A Mathematical Theory of Communication

The memorandum was published in 1948, but did not see widespread adoption until the publishing of the book with Warren Weaver who provided an introduction for a more general scientific audience, and included a small but poignant change in the title to "The Mathematical Theory of Communication".

Until Shannon, information was not a strictly defined technical term, but was associated with its everyday, overloaded meaning. He rather wanted to remove "meaning" from the definition and remove any semantic connotations, e.g. the language content of a message. He wrote that "the semantic aspects of communication are irrelevant to the engineering aspects".

In his explanation of information Weaver notes that: "information is a measure of one's freedom of choice when one selects a message". Meaning does not enter here as the message could very well be "pure nonsense". Information is closely associated with uncertainty: it can be thought of as a measure of surprise. Following up the word "White" with "House" is not surprising, and as such, "House" carries little information and is to some degree redundant. The same can be true for the many letters of the English alphabet, given their context. For example "_U cn prbly rd ths_". Shannon determined that English has a redundancy of about 50 percent.

The redundancy posits that certain sequences of symbols will be more likely than others, and that communication can be modeled as a _stochastic process_. The generation of messages is governed by a set of probabilities that depend on the state of the system and its history. In a simplified model, communication of language could be modeled as a Markov process, where the next symbol to appear depends solely on a number of symbols that preceded it.

Perhaps the most important contribution of Shannon's work is the introduction of entropy. According to Weaver, in the context of communication "information is exactly that which is known in thermodynamics as _entropy_". Entropy was introduced originally by Clausius in 1865, and later used by Boltzmann and Gibbs in his work in statistical mechanics. The entropy of a thermodynamical system is the measure of the number of states with significant probability of being occupied, multiplied by Boltzmann's constant.

Entropy then measures the uncertainty in a system, and that is in essence its information content. Shannon had originally planned to call this "_uncertainty_" but was dissuaded by Von Neumann[^1]:

> I thought of calling it "information", but the word was overly used, so I decided to call it "uncertainty". [...] Von Neumann told me, "You should call it entropy, for two reasons. In the first place your uncertainty function has been used in statistical mechanics under that name, so it already has a name. In the second place, and more important, nobody knows what entropy really is, so in a debate you will always have the advantage."

The diagram used by Shannon for his communication model bears many similarities with the one used for his cryptography paper, which allows us to see a clear path in the development of the ideas:

![A general communication system in [Shannon49a]](/assets/shannon_comm_channel.jpg)

_A general communication system in [Shannon49a]_

The entropy connects information to the amount of choice available when constructing messages, it measures the uncertainty involved in the "_selection of an event or how uncertain we are of the outcome_". When the probability of each symbol appearing is equal, one can use Nyquist's formula:

$$
H = n \log{S}.
$$

For the case where the probabilities of each symbol are determined by
$p_1...p_S$ the expression that Shannon defined was:

$$
H = -\sum_i{p_i log_2 p_i}
$$

Shannon defined a unit of measure for this information as "binary digits, or more _bits_", which he credited to John Tukey. A bit represents the unit of information that is present in the flipping of a coin, i.e. an event with two possible outcomes of equal probability.

From the concept of redundancy Shannon developed ways to communicate natural language more efficiently, by making use of the probabilities of different symbols. He further defined the channel capacity of a noisy channel (Shannon limit) and the possibility of perfect communication in a noisy channel through the noisy-channel coding theorem. Removing redundancy could increase the rate of transfer, which underpins the field of compression, while adding redundancy can enable correct communication in the presence of errors, the basis of coding theory.

The fundamental connection made was that information and probabilities were intrinsically connected: an event carries information related to the probability of observing it, as defined by Shannon's entropy.

As a more practical hardware example than the coin flip, Shannon noted that:

> A device with two stable positions, such as a flip-flop circuit, can store one bit of information. N such devices can store N bits, since the total number of possible states is $2^N$, and  $log_2 2^N = N$

At around the same time, in the same building that Shannon had developed his theory of information, the transistor had just been created.

## The discovery process

In our description of the development of information theory by Shannon, we have neglected some important previous work that better puts in context Shannon as one of the pillars of computer science. After graduating in 1936 with two bachelor's degrees in electrical engineering and mathematics from  the University of Michigan, Shannon joined MIT for his graduate studies. There, here worked on Vannevan Bush's mechanical computer, the differential analyzer [Bush36]. He spent his time analyzing the machine's circuits, and designing switching circuits based on Boole's algebra. In his 1937 master's degree thesis "A Symbolic Analysis of Relay And Switching Circuits" he proved that such circuits could be used to solve all problems that Boolean algebra could solve, which would later become the foundation of digital circuit design. The thesis has been called "possibly the most important, and the most noted, master's thesis of the century [Gardner87]".

At the advice of Bush, Shannon switched from electrical engineering to mathematics for his PhD studies, and suggested applying symbolic algebra to the problem of Mendelian genetics. Mendelian genetics was a branch of genetics that was met with resistance initially in the 1900s. Even after R.A. Fisher's seminal book "The Genetical Theory of Natural Selection" [Fisher30], Mendelian genetics were not clearly understood, particularly its basic components, the genes. Shannon's PhD thesis, "An Algebra For Theoretical Genetics", includes in the  introduction the following passage:

> Although all parts of the Mendelian genetics theory have not been incontestably established, still it is possible for our purposes, to act as though they were, since the results obtained are known to be the same _as if_ the simple representation which we give were true. Hereafter we shall speak therefore as though the genes actually exist and as though our simple representation of hereditary phenomena were really true, since so far as we are concerned, this might just as well be so.

It is important to take a moment and appreciate the workings of Shannon's approach here, and the liberty afforded to him to dig deep into a single idea, be it from the field of mathematics or the different era. Before his PhD dissertation Shannon published no articles other than his unpublished (though seminal) master's thesis. Although his Phd thesis was never published, (it sits at 45 citations according to Google scholar, compared to 105,460 for his information theory papers) he was given the chance to continue his research at the Institute of Advanced Study in Princeton, at that time occupied by giants such as Einstein and Kurt Gödel. There, he had the chance to discuss his ideas with mathematicians such as John Von Neumann, inventor of, among many other things, the computer architecture that underpins all modern computers.

It is safe to say that Shannon's interaction with many of the greatest thinkers of his era, Turing, Von Neumann, Einstein, Gödel, helped shape him as a scientist and enabled the development of one of the "great ideas" of the previous century.

Another important consideration is the fact that Shannon made his greatest work while employed at the labs of a private company, Bell Labs, either while being assigned work from the government on wartime efforts, or later on in his own work. It is remarkable to think that private companies would fund mathematicians to perform basic research, but that is exactly what Bell Labs was doing during that era. The lab, now owned by Nokia, counts 8 Nobel Prizes among its accomplishments [Bell18]. One may wonder if such a lab exists today. Private initiatives in machine learning like OpenAI may come close, but other efforts like Facebook's FAIR labs and Google's Deepmind are doing cutting edge research, but always with a product focus, the results of the research are expected to be, in some way, useful for business purposes.

The drive of Shannon is also an interesting topic. While looking at the cryptanalysis work it's easy to look back and draw a line between that and information theory, Shannon's personal process does not seem to reflect this. He is quoted in [Gleick11]:

> My mind wanders around, and I conceive of different things day and night. Like a science-fiction writer, I'm thinking, "What if it were like this?"

As is often the case then, curiosity, mixed with a brilliant mind and unrelenting rigor lead to the establishment of information theory. Indeed the needs were there: The inclusion of a noise source in Shannon's theory reflects his engineer self, rooted in the practicalities of communicating over imperfect channels, which is what his company, Bell (AT&T) required. But the theoretical foundation and mathematical rigor is what elevates this theory beyond simple applied science. It's the culmination of practical needs combined with the curiosity of a brilliant mind.

## Ongoing Impact

Ever since the publication of the book by Shannon and Weaver, information theory has been applied to pretty much every area in science to the point of Shannon calling out other researchers in his article "The Bandwagon" [Shannon56]. The Bandwagon is an interesting article on its own, and a rare one in the sense that it is the founder of a discipline chastising his pupils as it were for taking his theory, doing away with rigor and running away with it. It was a piece of writing, and scientific leadership that is rare in the current climate especially in areas full of hype, such as the current state of machine learning and "AI".

Nevertheless, information theory would prove critical in a variety of fields. That included neuroscience [Dimitrov11], biology [Adami04], economics [Maasoumi93], machine learning [MacKay03], cognitive science [Dretske81], linguistics and natural language processing [Harris91], and of course communication [Gallager68] and compression [Johnson03].

New applications are constantly being discovered as well. We note one example that is relevant to our research where coding theory is being used to speed up distributed machine learning [Lee18].

## Conclusions

The assignment description asks "How the invention changed our view of the world?". To this we would answer that before Shannon, information was a term with no clear meaning. Gleick uses the phrase "the 'stuff' of communication" and in the past "intelligence" was used to convey the content of a message. One of the most important contributions of Shannon might actually be doing away with "meaning", and focusing on what remains, ones and zeros, the presence of structure or not, a quantification of uncertainty that has lead to all the scientific advancement we see around us today.

With the ongoing development of machine learning, which in many ways has its roots set in information theory [MacKay03], and the promise of quantum computing [Nielsen10], the role of information theory will remain central in computer science and science in general. It is remarkable that one person can contribute so much to the development of science, but that is exactly what Shannon did.

[^1]: The validity of this story is challenged by [Gleick11]

## References

[Adami04] Adami, C. (2004). Information theory in molecular biology. Physics of Life Reviews, 1(1), 3-22.

[Bell18] https://www.bell-labs.com/about/recognition/, Retrieved 2018-04-26

[Bush36] Bush, Vannevar (1936). Instrumental Analysis. Bulletin of the American Mathematical Society. 42 (10): 649–69. doi:10.1090/S0002-9904-1936-06390-1

[Dimitrov11] Dimitrov, A. G., Lazar, A. A., & Victor, J. D. (2011). Information theory in neuroscience. Journal of Computational Neuroscience, 30(1), 1-5.

[Dretske81] Dretske, F. (1981). Knowledge and the Flow of Information.

[Fisher30] Fisher, R. A. (1930). The Genetical Theory of Natural Selection. The Clarendon Press.

[Gallager68] Gallager, R. G. (1968). Information theory and reliable communication (Vol. 2). New York: Wiley.

[Gardner87] Gardner, Howard (1987). The Mind's New Science: A History of the Cognitive Revolution. Basic Books. p. 144. ISBN 0-465-04635-5.

[Gleick11] Gleick, James (2011). The  Information: A History, A Theory, A Flood. Pantheon Books.

[Harris91] Harris, Z. (1991). Theory of language and information: a mathematical approach.

[Johnson03] Johnson Jr, P. D., Harris, G. A., & Hankerson, D. C. (2003). Introduction to information theory and data compression. CRC press.

[Lee18] Lee, K., Lam, M., Pedarsani, R., Papailiopoulos, D., & Ramchandran, K. (2018). Speeding up distributed machine learning using codes. IEEE Transactions on Information Theory, 64(3), 1514-1529.

[MacKay03] MacKay, D. J. (2003). Information Theory, Inference and Learning Algorithms. Cambridge University Press.

[MacKay52] MacKay, D. M., & McCulloch, W. S. (1952). The limiting information capacity of a neuronal link. Bulletin of Mathematical Biophysics, 14, 127–135.

[Nielsen10] Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information. Cambridge university press.

[Maasoumi93] Maasoumi , E. (1993). A compendium to information theory in economics and econometrics. Econometric reviews, 12(2), 137-181.

[Sloane98] Sloane, Neil (1998). Bibliography of Claude Elwood Shannon, http://neilsloane.com/doc/shannonbib.html, Retrieved 2018-04-26

[Shannon38] Shannon, Claude E. (1938). A Symbolic Analysis of Relay and Switching Circuits." Unpublished MS Thesis, Massachusetts Institute of Technology.

[Shannon40] Shannon, Claude E. (1938). An Algebra for Theoretical Genetics Ph.D. Thesis, Massachusetts Institute of Technology.

[Shannon48] Shannon, Claude E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, Vol. 27 (July and October 1948)

[Shannon49a] Shannon, Claude E.  (with Warren Weaver) The Mathematical Theory of Communication, University of Illinois Press, Urbana, IL, 1949. The section by Shannon is essentially identical to the previous item.

[Shannon49b] Communication Theory of Secrecy Systems, Bell System Technical Journal, Vol. 28 (1949), pp. 656-715. ``The material in this paper appeared originally in a confidential report `A Mathematical Theory of Cryptography', dated Sept. 1, 1945, which has now been declassified.'' Included in Part A.

[Shannon56] Shannon, Claude E. (1956). The Bandwagon. IRE Transactions on Information Theory, 2(1), 3.