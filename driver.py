#!/usr/bin/env python3

import sys
import os

import decoder
import extractor
import scorer
import gflags
import logger
FLAGS = gflags.FLAGS

if __name__ == '__main__':
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    import __main__
    mydir = os.path.dirname(__main__.__file__)

    if os.path.exists(FLAGS.rule_dump):
        logger.writeln('unfiltered rules exists: %s' % FLAGS.rule_dump)
    else:
        extractor.main()

    FLAGS.filter_file = FLAGS.input
    scorer.main()

    glue_grammar = os.path.join(mydir, 'test-extractor/monotonic_glue.gr')
    if FLAGS.persent is None:
        FLAGS.grammars = [FLAGS.filtered, glue_grammar]
    else:
        FLAGS.grammars = [FLAGS.persent, glue_grammar]

    decoder.main()
