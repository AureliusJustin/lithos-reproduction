typedef void(*fn)();
extern "C" __global__ void prelude(unsigned long long e){
    fn f=(fn)e;
    f();          // tail position, void->void, matching sig
}
