from PKLot2.input_data import *
from PKLot2.model import *
from PKLot2.mark_image import *
import tensorflow as tf
import random


def run_training():

    N_CLASSES = 2
    BATCH_SIZE = 64
    MAX_STEP = 20000
    learning_rate = 0.0001
    IMG_W = 28
    IMG_H = 28
    CHANNEL = 1
    is_train = 2

    tfrecords_path = '/home/bc/Documents/datasets/PKLot/tfrecords'
    train_tfrecords_name = 'train'
    val_tfrecords_name = 'val'
    logs_train_dir = './logs/train/'
    logs_val_dir = './logs/val/'
    img_path = '/home/bc/Documents/datasets/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_34_01.jpg'
    xml_path = '/home/bc/Documents/datasets/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_34_01.xml'


    # read data from tfrecords file
    train_batch, train_label_batch = read_and_decode(
                            os.path.join(tfrecords_path, train_tfrecords_name + '.tfrecords'), BATCH_SIZE)
    val_batch, val_label_batch = read_and_decode(os.path.join(tfrecords_path, train_tfrecords_name + '.tfrecords'),
                                                 BATCH_SIZE)
    # load the segmented images for testing
    segment_images, segment_labels = segment(img_path, xml_path)
    segment_images = tf.reshape(segment_images,[-1,28,28,1])

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, CHANNEL])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    train_logits = inference(x, BATCH_SIZE, N_CLASSES)
    train_loss = losses(train_logits, y_)
    train_op = training(train_loss, learning_rate)
    train_acc = evaluation(train_logits, y_)



    # save the model, only keep one version
    # create network or initialize network before initilizing the Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.group(tf.global_variables_initializer(),
        # tf.local_variables_initializer()))
        # sess.run(tf.local_variables_initializer())
        # sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            if is_train==1:

                max_acc = 0

                # starts to train
                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                        break
                    train_images, train_labels = sess.run([train_batch, train_label_batch])
                    _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                                    feed_dict={x:train_images, y_:train_labels})

                    if step % 50 == 0:
                        print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                        # summary_str = sess.run(summary_op)
                        # train_writer.add_summary(summary_str, step)

                    if step % 200 == 0 or (step + 1) == MAX_STEP:
                        val_images, val_labels = sess.run([val_batch, val_label_batch])
                        val_loss, val_acc = sess.run([train_loss, train_acc], feed_dict={x:val_images, y_:val_labels})
                        print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **' %(step, val_loss, val_acc*100.0))
                        # summary_str = sess.run(summary_op)
                        # val_writer.add_summary(summary_str, step)

                    if step % 500 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            elif is_train==2:
                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')
                test_images, test_labels = sess.run([segment_images, segment_labels])
                prediction = sess.run(train_logits, feed_dict={x: test_images[:BATCH_SIZE]})
                max_index = np.argmax(prediction, 1)
                print('prediction:', max_index)
                print('label:', test_labels[:BATCH_SIZE])
                print(sum(max_index==test_labels[:BATCH_SIZE])/BATCH_SIZE)

            else:
                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

                val_images, val_labels = sess.run([val_batch, val_label_batch])
                prediction = sess.run(train_logits, feed_dict={x: val_images})
                max_index = np.argmax(prediction, 1)
                print('prediction:', max_index)
                print('label:', val_labels)
                print(sum(max_index == val_labels[:BATCH_SIZE]) / BATCH_SIZE)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

# def evaluate_one_image():
#     BATCH_SIZE = 64
#     segment_images, segment_labels = segment_nparray(img_path, xml_path)
#     # rand = random.randint(0,9)
#     segment_images = segment_images[0:BATCH_SIZE]
#     segment_labels = segment_labels[0:BATCH_SIZE]   # 1
#
#     segment_images = segment_images.reshape(BATCH_SIZE, 28, 28, 1)
#
#     with tf.Graph().as_default():
#
#         N_CLASSES = 2
#
#         segment_images = tf.cast(segment_images, tf.float32)
#         logit = inference(segment_images, BATCH_SIZE, N_CLASSES)
#
#         logs_train_dir = './logs/train/'
#         saver = tf.train.Saver()
#
#         with tf.Session() as sess:
#             print("Reading checkpoints...")
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print('Loading success, global_step is %s' % global_step)
#             else:
#                 print('No checkpoint file found')
#
#             prediction = sess.run(logit)
#             max_index = np.argmax(prediction,1)
#             print('prediction: ',max_index)
#             print('segment_labels: ',segment_labels)
#             print('Accuracy: ',sum(max_index == segment_labels[:BATCH_SIZE]) / BATCH_SIZE)
#
#     print()
def evaluate_one_image():
    BATCH_SIZE = 64
    segment_images, segment_labels = segment(img_path, xml_path)
    # rand = random.randint(0,9)
    segment_images = segment_images[0:BATCH_SIZE]
    segment_labels = segment_labels[0:BATCH_SIZE]   # 1

    segment_images = tf.reshape(segment_images,[BATCH_SIZE, 28, 28, 1])

    with tf.Graph().as_default():

        with tf.Session() as sess:
            N_CLASSES = 2

            segment_images = tf.cast(segment_images, tf.float32)
            logit = inference(segment_images, BATCH_SIZE, N_CLASSES)

            logs_train_dir = './logs/train/'
            saver = tf.train.Saver()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit)
            max_index = np.argmax(prediction,1)
            print('prediction: ',max_index)
            print('segment_labels: ',segment_labels)
            print('Accuracy: ',sum(max_index == segment_labels[:BATCH_SIZE]) / BATCH_SIZE)

    print()
if __name__ == '__main__':
    # run_training()
    evaluate_one_image()