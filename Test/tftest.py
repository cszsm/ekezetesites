from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from sklearn.feature_extraction import DictVectorizer
import preparer
import helper

################

vectorizer = DictVectorizer()
vectorizer.fit(helper.generate_windows(4))

text = "A szélsőjobbos alt-right mozgalom Trump közelsége nem volt kérdéses, köszönhetően olyan lépéseknek, mint hogy Trump az alt-right egyik kedvelt lapjának, a Breitbartnak a vezérigazgatóját választotta kampányfőnökének és főstratégájának. Az ügy most vált különösen kényessé, amikor előkerült egy felvétel,amin az alt-right vezetője, a náci ideológiák terjesztése miatt Magyarországról is kitiltott Richard B. Spencer 'Hail Trump'-felkiáltással zárja egy alt-rightos konferenciáját. A résztvevők pedig náci karlendítéssel válaszoltak. Trump a felvételre úgy válaszolt: 'Megtagadom, és elítélem őket.' A Breitbart főnökének kritikájára pedig érzelmes húrokat pengetett meg. Bannont szerinte nagyon megviselte, hogy a nácikkal emlegették egy lapon a mainstream amerikai médiában. Sajátos logikával bizonyította, hogy Bannon nem szélsőjobbos: 'Ha azt gondolom, hogy ő náci, alt-rightos, vagy hasonló, akkor biztos hogy nem alkalmazom őt.' - érvelt. A Hollywood Reporter egyébként összeszedte, hogy mely más nyelvű lapok a Breitbart megfelelői. Szerintük a Kurucinfo a magyar Breitbart."
x_e, y_e = preparer.prepare_text(text, 4, "e")
vectorized_x = vectorizer.transform(x_e).toarray()

test = "Tévedsz. Eddig epedtek érte, hogy legyen, s nem volt, most majd a lelkük üdvösségét kínálnák, ha elmaradhatna, de nem tudjuk megakadályozni."
test_x, test_y = preparer.prepare_text(test, 4, "e")
vectorized_test = vectorizer.transform(test_x).toarray()

################

input_size = len(vectorized_x[0])
print(input_size)
output_size = 2

x = tf.placeholder(tf.float32, [None, input_size])

W = tf.Variable(tf.zeros([input_size, output_size]))
b = tf.Variable(tf.zeros([output_size]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, output_size])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


# for i in range(len(vectorized_x)):
sess.run(train_step, feed_dict={x: vectorized_x, y_: y_e})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(accuracy, feed_dict={x: vectorized_test, y_: test_y}))