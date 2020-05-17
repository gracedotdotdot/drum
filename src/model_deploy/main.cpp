#include "mbed.h"

#include <cmath>

#include "DA7212.h"

#include "accelerometer_handler.h"

#include "config.h"

#include "magic_wand_model_data.h"

// library for recognizing gesture

#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"
#include "uLCD_4DGL.h"

// #define bufferLength (32)

// #define signalLength (4096)

DA7212 audio;

int16_t waveform[kAudioTxBufferSize];

EventQueue queue(32 * EVENTS_EVENT_SIZE);

uLCD_4DGL uLCD(D1, D0, D2);

Thread t_DNN(osPriorityNormal, 120*1024); /*stacksize 120k*/

Thread t_show;

Thread t_music;

Serial pc(USBTX, USBRX);

// Set up logging.

  static tflite::MicroErrorReporter micro_error_reporter;

  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

char songList[3][30]={"Some thing just like this","there's no if", "heal the world"};
char modeList[3][10]={"Forward","Backward","Change"};

DigitalIn selectSwitch(SW2);
DigitalIn modeSwitch(SW3);
int cursor=0;
int song_cursor = 0;


float songFreq1[48];
float songFreq2[42];
float songFreq3[26];

float noteLength1[48]={1};
float noteLength2[42]={1};
float noteLength3[26]={1};


void playNote(int freq)

{
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  // the loop below will play the note for the duration of 1s
  for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
  {
    audio.spk.play(waveform, kAudioTxBufferSize);
  }
}


void showSong(){   
  //int num=0;
  while(selectSwitch==1 && modeSwitch==1){ //if mode switch is not pressed, show song list and current song playing
    //num=cursor; // if num=-1 ->detect cursor, else num equals to forward or backward
    uLCD.locate(1,1);
    for(int i=0; i<3;i++){
      if(i==cursor)
        uLCD.color(BLUE);
      else if(i==song_cursor){
        error_reporter->Report("show Song: song_cursor=%d=i=%d\n",song_cursor, i);
        uLCD.color(RED);
      }
      else
        uLCD.color(WHITE);
      for(int j=0; j<30 && songList[i][j]!='\0'; j++){
        uLCD.printf("%c", songList[i][j]);
      }
      uLCD.printf("\n");
    }
  }
  if(selectSwitch==0){
    //play music
    error_reporter->Report("show Song: play music\n");

  }

}
void showMode(){
  while(selectSwitch==1){
    uLCD.locate(1,1);
    //show info on ulcd
    for(int i=0; i<3;i++){
      if(i==cursor)
        uLCD.color(BLUE);
      else
        uLCD.color(WHITE);
      for(int j=0; j<30 && modeList[i][j]!='\0'; j++){
        uLCD.printf("%c", modeList[i][j]);
      }
      uLCD.printf("\n");
    }
  }
  
  //detect which selected mode
  switch(cursor){
    
    case 0: //forward
      //error_reporter->Report("Show Mode : song_cursor=%d\n",song_cursor);
      if(song_cursor==2) song_cursor=0;
      else song_cursor+=1;
      //error_reporter->Report("change to =%d\n",song_cursor);
      showSong();
      break;
    case 1: //backward
      if(song_cursor==0) song_cursor=2;
      else song_cursor-=1;
      showSong();
      break;
    case 2: // change song
      showSong(); 
      break; 
  }
}
void showInfo(void){
  while(1){
    if(modeSwitch==0){
      uLCD.cls();
      showMode();
    }else{
      showSong();
    }
    //wait(0.5);
  }
}


//Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;
  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }
 // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;
  return this_predict;
}

void DNN(void){
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;


  // The gesture index of the prediction
  int gesture_index;
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    //return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.

  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                             tflite::ops::micro::Register_RESHAPE(), 1);   
  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;
  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();
  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    //return -1;

  }
  int input_length = model_input->bytes / sizeof(float);
  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    //return -1;
  }
  error_reporter->Report("Set up successful...\n");

  while (true) {
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);
    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;
    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
      if(cursor>1) cursor=0;
      else cursor++;
      error_reporter->Report("cursor=%d\n",cursor);
    }

  }
}
// void loadSignal(void){
//   while(pc.readable()){
//     for(int i=0; i<48; i++){
//       songFreq1[i] = (float)atof(pc.getc());
//     }
//     // for(int i=0; i<42; i++)
//     //   songFreq2[i] = (float)atof(pc.getc());
//     // for(int i=0; i<26; i++)
//     //   songFreq3[i] = (float)atof(pc.getc());
//   }
//   pc.printf("end reading\n");
// }
//void loadSignalHandler(void) {queue.call(loadSignal);}
int main(int argc, char* argv[]) {
  //songFreq[i] = pc.getc();
  //loadSwitch.rise(queue.event(loadSignalHandler));
  // while(selectSwitch==1) {}
  // loadSignal();
  //char songFreq[48];
  //pc.baud(115200);
  while(pc.readable()){
    for(int i=0; i<48; i++){
      songFreq1[i] = (float)atof(pc.getc());
      
    }
  }
  for(int i=0; i<48; i++){
    pc.printf("signalFreq1[%d] =%f",i,songFreq1[i]) ;
    //pc.printf("signalFreq1[%d] =%c",i,songFreq[i]) ;
    wait(0.5);
  }
  wait(60);
  
  pc.printf("start dnn\n");
  t_DNN.start(DNN);
  pc.printf("start show mode\n");
  t_show.start(showInfo);
  pc.printf("start music\n");

  // t_music.start(callback(&queue, &EventQueue::dispatch_forever));
  

  // pc.printf("start t_music\n");
  // while(1){
  //   float *songFreq, *noteLength;
  //   pc.printf("song_cursor=%d\n",song_cursor);
  //   // switch(song_cursor){
  //   //   case 0: 
  //   //     //pc.printf("get song1==============================\r\n");
  //   //     songFreq = songFreq1; 
  //   //     noteLength = noteLength1;
  //   //     break;
  //   //   case 1: 
  //   //     songFreq = songFreq2; 
  //   //     noteLength = noteLength2;
  //   //     break;
  //   //   case 2: 
  //   //     songFreq = songFreq3; 
  //   //     noteLength = noteLength3;
  //   //     break;
  //   // }
    
  //   for(int i = 0; i < 42  && modeSwitch==1 ; i++)
  //   {
  //     pc.printf("start song for loop %d\r\n",i);
  //     if(modeSwitch==1){
  //       int length = noteLength1[i];
  //       while(length--)
  //       {
  //         queue.call(playNote, songFreq1[i]);
  //         if(length <= 1) wait(0.3);
  //       }
  //     }else{
  //       queue.cancel(0);
  //     }
  //   }
  //   pc.printf("end \r\n");

  // queue.cancel(0);
  //}

}

