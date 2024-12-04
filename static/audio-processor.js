class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 1024;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];
        
        if (input.length > 0) {
            const inputChannel = input[0];
            const outputChannel = output[0];
            
            for (let i = 0; i < inputChannel.length; i++) {
                outputChannel[i] = inputChannel[i];
                this.buffer[this.bufferIndex++] = inputChannel[i];
                
                if (this.bufferIndex >= this.bufferSize) {
                    this.port.postMessage(this.buffer.slice(0));
                    this.bufferIndex = 0;
                }
            }
        }
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor); 