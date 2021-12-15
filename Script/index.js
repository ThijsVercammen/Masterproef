// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)
        
        console.log('--- 1');
        /*
        var file = new File('./img.jpg')
        console.log('--- 1');
        var reader = new FileReader();
        console.log('--- 1');
        reader.readAsArrayBuffer(file);
        console.log('--- 1');
        console.log(reader.result());
        console.log('--- 2');
        // prepare inputs. a tensor need its corresponding TypedArray as data
        
        */
        const fs = require('fs');
      //  const dataA = null;
        const buf = fs.readFileSync('./img.jpg');
        console.log('--- 2');
       // dataA = Float32Array.from(buf.buffer);

        //var array = new ArrayBuffer(data);
        console.log('--- 2');
        //const dataA = Float32Array.from(array);
        const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        const session = await ort.InferenceSession.create('./onnx_model.onnx');
        console.log('--- 3');
        const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
        const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

        // prepare feeds. use model input names as keys.
       // const feeds = { a: tensorA, b: tensorB};

        // feed inputs and run
        const results = await session.run({ a: tensorA, b: tensorB});

        // read from results
        const dataC = results.c.data;
        console.log(`data of result tensor 'c': ${dataC}`);

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();
