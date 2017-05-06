public class Bridge {
    static {
        System.loadLibrary("bridge");
    }

    private native String recognize(float image[], int length);
}
