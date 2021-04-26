# -----------------------------------------------------------------------------
# Florence Gurney-Cattino, Jameson Albers, Zongrui Liu
# CS 5002, Spring 2021
# Final Project: Multinomative Naive Bayes Classifier
#
# This program includes functions to import song lyrics and genre from a 
# dataset, process them, train a Bayes classifier, and output a prediction list.
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

song_1 = 'paint lips turn head need help zip dress cash unattached word sound like song gotta cause need reset deep breath yeah think time heart break late solo saturday night go plus fall pickup line single summer rebound lover tie sleep like queen california think time heart break think time heart break little phase little space put poker face givin rest think time heart break late solo saturday night go plus fall pickup line single summer rebound lover tie sleep like queen california think time heart break think time heart break long overdue think time heart break single summer rebound think time heart break think time heart break late solo saturday night go plus fall pickup line single summer rebound lover tie sleep like queen california think time heart break think time heart break'

song_2 ='drive drive night say leavin crack lyin heart go pack altima gas dash tryin damnin tear hold cause california coast dream hollywoods hill beach backroads sunset boulevard california breakin heart waitress waitin ihop throw universal city glamour shoot pocket hopin somebody think pretty time tomorrow colorado drownin sorrow bottle beam hit flagstaff maybe backtrack wouldn cause california coast dream hollywoods hill beach backroads sunset boulevard california breakin heart time california coast dream hollywoods hill beach backroads sunset boulevard california breakin heart california breakin heart breakin heart'

song_3 = 'yesterday come go today deal hand heartaches play say mean different things word ring time keep raise stake tonight dream tomorrow feel fade away go tear fill pain fall like rain constant reminders grey wait shine hour darkest time moment crime moment crime live forget free baby come baby come grey wait shine hour darkest time moment crime moment crime time moment crime'

song_4 = 'yeah south blood crimson yeah blue jeans fade greasestained hand yeah scar knuckle yeah come disrespectin homeland homeland yeah yeah shit go thousand shoot round railroad cover grind barb wire fence wrap house fourwheelers dirt bike model high rise snipers spittin beech hilltop rise yeah motherfucker everybody team bikers klan members police everybody figurin news shit purpose clue white live black live blue matter point stand group cause people tryna kill cause fuckin beliefs towelheads subway bomb brief bullshit stand people motherfuckin throats bleed equal yeah south blood crimson blue jeans fade greasestained hand scar knuckle come disrespectin homeland homeland yeah thunder runnin curtain fuck know attitude certain tryna purpose vultures head tellin things darn prove friends fallin face earth cause know somethin unity bustin southern motherfucker real live home flag steel gunpowder kerosene grenades pride tryna harm country ride bury bitch throw smell sweet freedom yeah south blood crimson blue jeans fade greasestained hand scar knuckle come disrespectin homeland homeland yeah homeland homeland homeland yeah homeland yeah homeland homeland yeah homeland yeah homeland yeah south blood crimson blue jeans fade greasestained hand scar knuckle come disrespectin homeland yeah homeland homeland'

song_5 = 'drink whiskey time hold women fine little money blow dime tryin untangle mind choices wrong good lord spend nights edge stumble cross line tryin untangle mind lonesome stone devil lookin high try untangle mind know heartache ticket guess fine tryin untangle mind lonesome stone devil lookin high tryin untangle mind tryin untangle mind'

song_6 = 'whoo come black mibs come mibs come black remember good guy dress black remember face face contact title hold mean think blink go black suit black raybans walk silence guard violence government list straight exist name fingerprint somethin strange watch cause know mibs come black black galaxy defenders oooh oooh oooh come black black remember remember deepest darkest night horizon bright light enter sight tight cameras zoom impendin doom like boom black suit room quickness talk witness hypnotizer neuralyzer vivid memories turn fantasy mibs kick yaknahmean noisy cricket wicked line defense worst scum universe fear cheer near jeer fearless mibs freezin flack stand black black black bounce bounce bounce bounce slide slide slide slide slide slide slide walk walk walk walk neck work freeze come black black galaxy defenders oooh oooh oooh come black black remember alright check tell closin know imposin trust section believe protection cause things need place need witcha life forget roswell crap black suit cause come black come galaxy defenders galaxy defenders come black come remember remember come black come galaxy defenders oooh oooh oooh come black remember'

song_7 = 'work niggas beef gonna spray pump street nigga want beef gonna spray bronco bone skin fish folay wire real kanye turn boys wine yayo cause grid paint picture rhyme songs like movies play nigga want beef gonna spray swiss cheese like nose mix bread bitch dudes crew strangle string doorag lyric commercial'

song_8 = 'sink fuck spine second guess crime snort slug cross fuck line banknorthside coffin ride basquiat trapaholics mixtapes drop shit fuck niggas grey sign rough diamonds tryna shine uiuicide know know know dive head crucify lucifer cry tell choose noose knife knife lyric commercial'

song_9 = 'fuckin cockroaches motherfuckin freebandz want wanna play cartel nigga montana montana montana montana check ears montana montana montana bout porsche montana montana montana leave choice montana montana montana streets fresh banana boat come straight east niggas split canteloupe tell wanna meet come gang dope cigar loud lace fuck porsche carrera panamera dash drop cash gutta death stick recipe lyric commercial'

song_10 = 'spit shit damn nigga ridiculous nigga lose inconspicuous incognito niggas ready flow nigga know spit deadly fear dead street hole ghettos gradually disaster tear laughter gonna style touchin nigga wipe obvious lie dead wise guy bitch niggaz feel fuckin break long wrong dead go bomb clarify vain think motherfuckers playin baby lord deny long alive want piece respect till date demise baby kill thinkin game playin operatin like plan baby kill whatcha wanna lyric commercial'

country_lyrics = [song_1.split(), song_2.split(), song_3.split(), song_4.split(), song_5.split()]

hiphop_lyrics = [song_6.split(), song_7.split(), song_8.split(), song_9.split(), song_10.split()]

total_words = set()

for song in country_lyrics:
    for word in song:
        total_words.add(word)

for song in hiphop_lyrics:
    for word in song:
        total_words.add(word)

data = []

for twang in country_lyrics:
    x = dict()
    for word in twang:
        if word in x:
            x[word] += 1
        else:
            x[word] = 1
    data.append(x)
        
for rhyme in hiphop_lyrics:
    x = dict()
    for word in rhyme:
        if word in x:
            x[word] += 1
        else:
            x[word] = 1
    data.append(x)

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(data)
Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
mnb = MultinomialNB()
mnb.fit(X,Y)


test_country = 'friends move time see smile go memory pretty picture hang wall compare fault know fall believe girl try memory good best tell compete ocean boulevard heart close eye place feel like home cause memory memory close smell perfume time hold hold right mind memory good best tell compete ocean boulevard heart close eye place feel like home cause memory memory know fall believe girl try memory good best tell compete ocean boulevard heart close eye place feel like home cause memory memory memory memory'

test_hiphop = 'fatal aspects crackin nigga movement look like circus look clown funny hat rap lions bear scarecrows gimmicks act crack strap jail best believe fatskios snitch niggas ceos cop glocks peanut brain pistachios trashy hoe jump like trapeze artists niggas like lookin like look artists universal circus fulla clown cowards kiss niggas lips nose open drown powder world girls look like thugs curl pearl sex look like hug tell travel group usually handbags lipgloss shade tight jeans lookin like parade award radio ticket circus niggas line kick nigga nigga nigga circus clown clown bozos flow homos need train like mothafuckin dojo flow cold mojo slow think hard know possin like photo crimes twist bitch vicious suspicious instance check lyric sheet somethin write lyric think catchy suspicious'

test_songs = [test_country.split(), test_hiphop.split()]

test_data = []

for song in test_songs:
    x = dict()
    for word in song:
        if word in total_words:
            if word in x:
                x[word] += 1
            else:
                x[word] = 1
    for ref_word in total_words:
        if ref_word not in x:
            x[ref_word] = 0
    test_data.append(x)

for i in range(2):
    print(len(test_data[i]))


X_test = dv.fit_transform(test_data)

print(mnb.predict(X_test))