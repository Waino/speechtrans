#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse

from onmt.translate.E2ETranslator import make_e2e_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts


def main(opt):
    translator = make_e2e_translator(opt, report_score=True)
    translator.translate(opt.src, opt.tgt, opt.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate_e2e.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
