import os

from coverage import CoveragePlugin, FileReporter, FileTracer


class LispCoverage(CoveragePlugin):

    def file_tracer(self, filename):
        _, ext = os.path.splitext(filename)
        if ext == ".lpy":
            return LispTracer(filename)
        return None

    def file_reporter(self, filename):
        return LispReporter(filename)

    def find_executable_files(self, src_dir):
        for dirpath, dirnames, filenames in os.walk(src_dir):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext == ".lpy":
                    yield os.path.join(dirpath, filename)

    def sys_info(self):
        return super().sys_info()


class LispTracer(FileTracer):
    __slots__ = ("_filename",)

    def __init__(self, filename: str):
        self._filename = filename

    def source_filename(self):
        return self._filename

    def has_dynamic_source_filename(self):
        return False

    def line_number_range(self, frame):
        return super().line_number_range(frame)


class LispReporter(FileReporter):

    def lines(self):
        with open(self.filename) as f:
            return set([i for i, _ in enumerate(f.readlines())])

    def excluded_lines(self):
        return super().excluded_lines()

    def translate_lines(self, lines):
        return super().translate_lines(lines)

    def arcs(self):
        return super().arcs()

    def no_branch_lines(self):
        return super().no_branch_lines()

    def translate_arcs(self, arcs):
        return super().translate_arcs(arcs)

    def exit_counts(self):
        return super().exit_counts()

    def missing_arc_description(self, start, end, executed_arcs=None):
        return super().missing_arc_description(start, end, executed_arcs)

    def source_token_lines(self):
        return super().source_token_lines()


def coverage_init(reg, options):
    reg.add_file_tracer(LispCoverage())
