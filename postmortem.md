# Post-mortem of end-to-end speech translation attempt

## Useful parts

LasEncoder in [onmt/modules/ListenAttendSpell.py](https://github.com/Waino/speechtrans/blob/master/onmt/modules/ListenAttendSpell.py).
This is the "Listener" from the Listen-Attend-Spell (LAS) model

Chan, William, et al. 2016
"Listen, attend and spell: A neural network for large vocabulary conversational speech recognition." 
Acoustics, Speech and Signal Processing (ICASSP).

Kaldi feature input
tools/data\_preprocess/\*
and some parts of onmt/io/E2EDataset.py


## Lessons learned

### torchtext

Torchtext, the library used by OpenNMT-py for numerizing textual data, suffers from poor usability.
The interfaces are unintuitive, poorly documented, and do not support introspection in the REPL.
Torchtext requires the entire dataset to be fed into the data set constructor.
Torchtext does not natively support partial loading of very large data sets,
which has lead OpenNMT-py to work around this deficiency
in a way that fights the library rather than use it.

Despite this, it was not a good decision to try to circumvent torchtext.
The consequences of our data loading scheme rippled through the system causing many problems.
Using the torchtext RawField (which we noticed too late) could have resulted in a cleaner system.

### Transformer

Different encoder and decoder architectures in OpenNMT-py are not easily compatible with each other.
[https://github.com/OpenNMT/OpenNMT-py/projects/4](https://github.com/OpenNMT/OpenNMT-py/projects/4)

In particular, the RNN modules are not compatible with the Transformer modules.
There are at least 2 points of incompatibility:

* The initial states required by the RNN
* The mask computations
        (torchtext produces sequence lengths. RNN uses them to compute a mask.
        Transformer instead computes a mask directly based on the padding index.)

The architecture would have been easier to implement if we would have chosen RNN instead of Transformer.
Alternatively the mask computations could have been reimplemented in a common way
and contributed back to OpenNMT-py.

### Multi-loss MTL

Our system had two inputs (English audio and English text) and two outputs (English text and German text).
We wanted to encode an input and then decode both outputs.
This results in a Y-shaped computational graph, with one active encoder, both decoders,
and two separate losses.

Implementing this system was complex.
Performing the tasks one by one would waste computation, but would have been much easier to implement.

### Overloaded src/trg

The meaning of "source" and "target" was overloaded in our system,
alternatively meaning languages (source: English, target: German),
or input and output of the network.

Confusion between these two meanings lead to unnecessarily complex code.
