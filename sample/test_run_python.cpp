#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cerr << "Args Err!" << std::endl;
		std::cerr << "Usage:" << argv[0] << " <script>  <args> <...>" << std::endl;
		return -1;
	}

	pid_t child = fork();
        if (child == -1) {
                std::cout << "[cluster] create cluster process failed.\n" << std::endl;
                return -1;
        } else if (child == 0){
		std::cout << "child process!" << std::endl;
                const char* shell = "python";
                execvp(shell, argv);
        }

	int status = 0;
	waitpid(child, &status, 0);
	std::cout << "main process exit!" << std::endl;

	return 0;
}
