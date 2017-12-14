# Scratchpad
A place to jot down ideas.

## Novel ideas
- Deeper networks
- Wider networks
- Different conv filter sizes
- Noisy inputs
- Dropout

## Some notes about data preprocessing
In the original code, data is preprocessed from raw .wav files into numpy arrays containing the magnitude and phase spectrograms from the short time Fourier transfrom (STFT). The class `LargeDataset` in `dataset.py` handles the transformation and subsequent batch handling during network training. How does it work?

1. The features (i.e. spectrograms) from the audio files are saved in a directory. If using the `compute_transform` function found in class `Transform` in `transform.py`, then the features for each audio file should be in this directory as:
    - {filename}\_\_{m,p}\_.data : numpy array containing the magnitude (m) or phase (p) spectrogram
    - {filename}\_\_{m,p}\_.shape : binary file containing shape of array
2. The LargeDataset class is pointed to the feature directory via the `path_transform_in` argument in the constructor.
3. It calls updatePath to update its list of .data files (self.file_list), which are all the .data files in the feature directory.
4. updatePath also updates the cumulative number of points in the file list (`self.num_points`) and total points (`self.total_points`), where a point is a time window of size `time_context`. This is done via the getNum function, which essentially return the `np.floor(time_axis / time_context`, plus a term if using overlap:

```python
def getNum(self,id):
        """
        For a single .data file computes the number of examples of size \"time_context\" that can be created
        """
        shape = self.get_shape(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace('.data','.shape')))
        time_axis = shape[1]
        return np.maximum(1,int(np.floor((time_axis + (np.floor(float(time_axis)/self.time_context) * self.overlap))  / self.time_context)))
```
5. updatePath also updates the input and output feature sizes via the `getFeatureSize` function, which returns the number of features (self.input_size) and number of features * number of sources (self.output_size) for each .data file.
6. Finally, updatePath calls `initBatches()`, which allocates memory needed for output. Several class variables are set:

```python
self.batch_size = np.minimum(self.batch_size,self.num_points[-1]) # size of each batch
self.iteration_size = int(self.total_points / self.batch_size)    # number of batches in dataset
self.batch_memory = np.minimum(self.batch_memory,self.iteration_size) # minimum number of batches to load into memory
#...
self.batch_inputs = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.input_size), dtype=self.tensortype)
        self.batch_outputs = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.output_size), dtype=self.tensortype)
```
7. At this point, all files and directories are accounted for, but nothing has actually been  loaded into memory. `initBatches()` calls `loadBatches()`, which loads batches into `self.batch_inputs` and `self.batch_outputs` once the current store is exhausted, by itself calling `genBatches()`. First, `genBatches()` calls `getNextIndex()` to update class variables that set the time window for the next batch:

```python
def getNextIndex(self):
    """
    Returns how many batches/sequences to load from each .data file
    """
    # next time point = (# of loads into memory) * (# of time points per load)
    target_value = (self.scratch_index+1)*(self.batch_memory*self.batch_size)
    # next file index = right-sided search of files with cumulative sum = next time point
    idx_target = np.searchsorted(self.num_points,target_value, side='right')
    # End case: set idxend to the number of points in the last file, and nindex
    # to the last file
    if target_value>self.num_points[-1] or idx_target>=len(self.num_points):
        idx_target = idx_target - 2
        target_value = self.num_points[idx_target]
        self.idxend = self.num_points[idx_target] - self.num_points[idx_target-1]
        self.nindex = idx_target
    # Otherwise, set idxend to number of points after file ending just prior to target
    # time point, and nindex to that file
    else:
        while target_value<=self.num_points[idx_target]:
            idx_target = idx_target - 1
        self.idxend = target_value - self.num_points[idx_target]
        self.nindex = idx_target
```
8. Next, `genBatches()` decides how much to load from how many files in order to produce batches of the desired length. It utilizes the `loadFile()` function to load .data files into memory between indices `idxbegin` and `idxend`.  As seen in the `loadInputOutput()` helper function, the .data file `file` contains the input, or unseparated STFT, in `allmixinput = file[0]` and the output, or source separated STFTs, in `allmixoutput = file[1:]`. The STFTs are scaled by a (log) scale factor:

```python
#apply a scaled log10(1+value) function to make sure larger values are eliminated
#bach10 training: mult_factor_in = mult_factor_out = 0.3 (0.2 for testing)
#                 log_in = log_out = False
if self.log_in==True:
    allmixinput = self.mult_factor_in*np.log10(1.0+allmixinput)
else:
    allmixinput = self.mult_factor_in*allmixinput
if self.log_out==True:
    allmixoutput = self.mult_factor_out*np.log10(1.0+allmixoutput)
else:
    allmixoutput = self.mult_factor_out*allmixoutput
```
9. The inputs and outputs in `loadFile()` are originally set via `loadOutput()` to:

```python
size = idxend - idxbegin
inp = np.zeros((size, self.time_context, self.input_size), dtype=self.tensortype)
out = np.zeros((size, self.time_context, self.output_size), dtype=self.tensortype)
```
10. If the file size is smaller than `time_context`, then the first part of `inputs` and `outputs` are taken from the file:

```python
if self.time_context > allmixinput.shape[1]:
    inputs[0,:allmixinput.shape[1],:] = allmixinput[0]
    outputs[0, :allmixoutput.shape[1], :allmixoutput.shape[-1]] = allmixoutput[0]
    # ...
    # concatenate features from rest of sources to third (feature) dimension
    for j in range(1,self.nsources):
        outputs[0, :allmixoutput.shape[1], j*allmixoutput.shape[-1]:(j+1)*allmixoutput.shape[-1]] = allmixoutput[j]
```
10. Otherwise, samples of size `time_context` are taken from the file along the time dimension until the target number of loaded samples is satisfied:

```python
else:
    while (start + self.time_context) < allmixinput.shape[1]:
        if i>=idxbegin and i<idxend:
            # separate variables names for memory clearing
            allminput = allmixinput[:,start:start+self.time_context,:] #truncate on time axis so it would match the actual context
            allmoutput = allmixoutput[:,start:start+self.time_context,:]
            inputs[i-idxbegin] = allminput[0]
            outputs[i-idxbegin, :, :allmoutput.shape[-1]] = allmoutput[0]
            # ...
            # concatenate features from rest of sources to third (feature) dimension
            for j in range(1,self.nsources):
                outputs[i-idxbegin,:, j*allmoutput.shape[-1]:(j+1)*allmoutput.shape[-1]] = allmoutput[j,:,:]
            # ...

        i = i + 1
        start = start - self.overlap + self.time_context
        #clear memory
        allminput=None
        allmoutput=None
```
11. `loadFile()` returns a dictionary of input and output (and other) values to `genBatches()`. After smartly loading from the correct number of files in sequence to fill `self.batch_inputs` and `self.batch_outputs`, the batches are shuffled via `shuffleBatches()`. Finally, class variables are incremented accordingly in anticipation of the next call.