MAP=config.txt
DEBUGGER= -m pdb

MAIN=call_me_maybe.py

TOCLEAN=.mypy_cache

MYPYFLAGS=--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

all: run

pyproject.toml:
	uv init

install: pyproject.toml requirements.txt
	uv add -r requirements.txt
	uv sync

run: install
	$(PYTHON) $(MAIN)

clean:
	py3clean .
	rm -rf $(TOCLEAN)

lint:
	flake8 MazeGen a_maze_ing.py setup.py
	mypy MazeGen a_maze_ing.py  $(MYPYFLAGS)

lint-strict:
	flake8 MazeGen a_maze_ing.py setup.py
	mypy MazeGen a_maze_ing.py --strict --ignore-missing-imports

SILENT: all install run debug build install clean lint lint-strict