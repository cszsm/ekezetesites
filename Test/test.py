from sklearn.feature_extraction import DictVectorizer
import preparer
import normalizer

vectorizer = DictVectorizer()

text = "Tévedsz. Eddig epedtek érte, hogy legyen, s nem volt, most majd a lelkük üdvösségét kínálnák, ha elmaradhatna, de nem tudjuk megakadályozni."
text2 = "1871. március 25-én hangversenyt adott a Vigadóban Reményi Ede és Olga Janina nevű tanítványa közreműködésével. Pesten az első állandó lakása a Szél, ma Nádor utca 20. szám alatt, a második emeleten volt (1871. szeptember 16-ától december 16-áig).[105] Olga Janina, az önjelölt „kozák grófnő” valójában sem gróf, sem kozák nem volt, valódi neve Olga Zielinska, aki egy meggazdagodott lengyel középosztálybeli családból származott. Szabados életet élt, kábítószerfüggő, kiegyensúlyozatlan, gazdag nő volt. Liszttel 1869-ben ismerkedett meg Rómában, ezután több helyre, így Magyarországra is meghívatlanul követte, ahol tanítványául szegődött. Pesten revolverrel fenyegette meg a művészt, majd öngyilkos akart lenni. Liszt barátai később azzal fenyegették meg a nőt, hogy ha nem hagyja békén Lisztet, a rendőrséggel fog meggyűlni a baja. Ekkor Janina Párizsba ment, és álnéven egy lejárató könyvet jelentetett meg a művészről (Souvenirs d’une Cosaque).[106]"
text3 = "A szélsőjobbos alt-right mozgalom Trump közelsége nem volt kérdéses, köszönhetően olyan lépéseknek, mint hogy Trump az alt-right egyik kedvelt lapjának, a Breitbartnak a vezérigazgatóját választotta kampányfőnökének és főstratégájának. Az ügy most vált különösen kényessé, amikor előkerült egy felvétel,amin az alt-right vezetője, a náci ideológiák terjesztése miatt Magyarországról is kitiltott Richard B. Spencer 'Hail Trump'-felkiáltással zárja egy alt-rightos konferenciáját. A résztvevők pedig náci karlendítéssel válaszoltak. Trump a felvételre úgy válaszolt: 'Megtagadom, és elítélem őket.' A Breitbart főnökének kritikájára pedig érzelmes húrokat pengetett meg. Bannont szerinte nagyon megviselte, hogy a nácikkal emlegették egy lapon a mainstream amerikai médiában. Sajátos logikával bizonyította, hogy Bannon nem szélsőjobbos: 'Ha azt gondolom, hogy ő náci, alt-rightos, vagy hasonló, akkor biztos hogy nem alkalmazom őt.' - érvelt. A Hollywood Reporter egyébként összeszedte, hogy mely más nyelvű lapok a Breitbart megfelelői. Szerintük a Kurucinfo a magyar Breitbart."

f = open("text.txt")
text4 = f.read()


textn = normalizer.normalize_text(text)
text2n = normalizer.normalize_text(text2)
text3n = normalizer.normalize_text(text3)
text4n = normalizer.normalize_text(text4)


print(textn)

print(((preparer.prepare_text(text, 2, "e"))[2]))
# print(len(vectorizer.fit_transform(((preparer.prepare_text(text2, 4, "e")))).toarray()[0]))


# print(((helper.prepare_text(textn, 2, "e"))[0]))
# print(len(vectorizer.fit_transform(((preparer.prepare_text(textn, 2, "e")))).toarray()[0]))
# print(len(vectorizer.fit_transform(((preparer.prepare_text(text2n, 2, "e")))).toarray()[0]))
# print(len(vectorizer.fit_transform(((preparer.prepare_text(text3n, 2, "e")))).toarray()[0]))
# print(len(vectorizer.fit_transform(((preparer.prepare_text(text4n, 2, "e")))).toarray()[0]))

# print(vectorizer.get_feature_names())
