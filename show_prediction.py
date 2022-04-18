import matplotlib.patches as mpatches

from model_predict import *


def show_image_prediction(test_ids, X_test, Y_test, Y_pred):
    plot1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle("Segmentation of " + test_ids[0])
    ax1.imshow(X_test[0])
    ax2.imshow(np.squeeze(Y_test[0]))
    ax3.imshow(np.squeeze(Y_pred[0]))
    ax1.set_title("X_train")
    ax2.set_title("Y_train")
    ax3.set_title("Y_predicted")
    plt.show()


def show_confusion_matrix_img(test_ids, X_test, Y_test, Y_pred):
    sum = Y_pred[0] + Y_test[0]
    differ = Y_pred[0] - Y_test[0]
    img = X_test[0]
    color = np.zeros((img.shape), dtype=np.uint8)
    color[differ == 1] = np.array([219, 62, 50])
    color[differ == -1] = np.array([83, 161, 67])
    color[sum == 2] = np.array([31, 117, 204])

    img[sum != 0] = cv2.addWeighted(img[sum != 0], 0.6, color[sum != 0], 0.4, 0)

    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    patches = [mpatches.Patch(color="#db3e32", label="FALSE POSITIVE"),
               mpatches.Patch(color="#53a143", label="FALSE NEGATIVE"),
               mpatches.Patch(color="#1f75cc", label="TRUE POSITIVE")]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Confusion matrix for " + test_ids[0])
    plt.show()


def show_plot(architecture):
    if architecture == 'U-Net':
        model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    elif architecture == 'FCN':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    else:
        return

    test_ids, X_test, Y_test, Y_pred = predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, THRESHOLD)
    if task == '1':
        show_image_prediction(test_ids, X_test, Y_test, Y_pred)
    if task == '2':
        show_confusion_matrix_img(test_ids, X_test, Y_test, Y_pred)


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default='1',
                        help='1 - show image prediction, 2 - show confusion matrix on prediction')
    parser.add_argument('--architecture',
                        default='U-Net',
                        help='Possible: U-Net, FCN')
    parser.add_argument('--THRESHOLD',
                        default=0.3)
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--MASK_TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASKS PATH')
    parser.add_argument("--WEIGHTS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\U_Net_data\u-net_model_epoch=10_valloss=0.1094.h5")
    parser.add_argument("--IMG_PATH",
                        default=r"5F781B80-71AA-4BE3-ABD7-0198659685C7.jpg")

    args = parser.parse_args()
    task = args.task
    architecture = args.architecture
    THRESHOLD = args.THRESHOLD
    TEST_PATH = args.TEST_PATH
    MASK_TEST_PATH = args.MASK_TEST_PATH
    WEIGHTS_PATH = args.WEIGHTS_PATH
    IMG_PATH = args.IMG_PATH

    show_plot(architecture)
