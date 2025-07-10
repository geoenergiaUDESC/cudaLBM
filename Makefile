include common.mk

EXECUTABLE = momentBasedD3Q19
SOURCE = LBM.cu

default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_FLAGS) $(SOURCE) -o $@

install: clean uninstall $(EXECUTABLE)
	cp $(EXECUTABLE) build/bin/
	rm -f $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

uninstall:
	rm -f build/bin/$(EXECUTABLE)