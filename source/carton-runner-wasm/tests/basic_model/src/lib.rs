#[no_mangle]
#[used]
pub static mut INPUT1: [f32; 20] = [0f32; 20];

#[no_mangle]
#[used]
pub static mut INPUT2: [f32; 20] = [0f32; 20];

#[no_mangle]
#[used]
pub static mut OUTPUT: [f32; 20] = [0f32; 20];

#[no_mangle]
pub extern "C" fn infer() {
    unsafe {
        for i in 0..20 {
            OUTPUT[i] = INPUT1[i] + INPUT2[i];
        }
    }
}
