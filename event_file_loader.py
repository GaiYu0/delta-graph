# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functionality for loading events from a record file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf


class EventFileLoader(object):
  """An EventLoader is an iterator that yields Event protos."""

  def __init__(self, file_path):
    if file_path is None:
      raise ValueError('A file path is required')
    file_path = tf.compat.v1.resource_loader.readahead_file_path(file_path)
    tf.compat.v1.logging.debug('Opening a record reader pointing at %s', file_path)
    with  tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
      self._reader = tf.pywrap_tensorflow.PyRecordReader_New(
          tf.compat.as_bytes(file_path), 0, tf.compat.as_bytes(''), status)
    # Store it for logging purposes.
    self._file_path = file_path
    if not self._reader:
      raise IOError('Failed to open a record reader pointing to %s' % file_path)

  def Load(self):
    """Loads all new values from disk.

    Calling Load multiple times in a row will not 'drop' events as long as the
    return value is not iterated over.

    Yields:
      All values that were written to disk that have not been yielded yet.
    """
    tf.compat.v1.logging.debug('Loading events from %s', self._file_path)
    while True:
      try:
        if not inspect.getargspec(self._reader.GetNext).args[1:]: # pylint: disable=deprecated-method
          self._reader.GetNext()
        else:
          # GetNext() expects a status argument on TF <= 1.7
          with tf.errors.raise_exception_on_not_ok_status() as status:
            self._reader.GetNext(status)
      except (tf.errors.DataLossError, tf.errors.OutOfRangeError) as e:
        tf.compat.v1.logging.debug('Cannot read more events: %s', e)
        # We ignore partial read exceptions, because a record may be truncated.
        # PyRecordReader holds the offset prior to the failed read, so retrying
        # will succeed.
        break
      event = tf.compat.v1.Event()
      event.ParseFromString(self._reader.record())
      yield event
    tf.compat.v1.logging.debug('No more events in %s', self._file_path)


def main(argv):
  if len(argv) != 2:
    print('Usage: event_file_loader <path-to-the-recordio-file>')
    return 1
  loader = EventFileLoader(argv[1])
  for event in loader.Load():
    print(event)
  return 0


if __name__ == '__main__':
  tf.app.run()
