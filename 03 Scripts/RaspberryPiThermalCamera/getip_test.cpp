#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <linux/if_link.h>
#include <iostream>
#include <string.h>

int getRPiAddress(const char*& address)
{
    struct ifaddrs *ifaddr, *ifa;
    int family, s, n;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
	perror("getifaddrs");
	exit(EXIT_FAILURE);
    }

/* Walk through linked list, maintaining head pointer so we
   can free list later */
    for (ifa = ifaddr, n = 0; ifa != NULL; ifa = ifa->ifa_next, n++) {
	if (ifa->ifa_addr == NULL)
	    continue;

	family = ifa->ifa_addr->sa_family;
        // the pi address for local networks is stored under wlan0
	if(strcmp(ifa->ifa_name,"wlan0")==0){
            // there are versions of wlan0, AF_INET and AF_INTET6
            // we want the AF_INET one
	    if (family == AF_INET) {
	         s = getnameinfo(ifa->ifa_addr,
		    (family == AF_INET) ? sizeof(struct sockaddr_in) :
		    sizeof(struct sockaddr_in6),
		    host, NI_MAXHOST,
		    NULL, 0, NI_NUMERICHOST);
		    // if failed to get name info, print error and return 1
		    if (s != 0) {
			std::cout << "getnameinfo() failed: %s\n" << gai_strerror(s) << std::endl;
			return 1;
		    }
                    // update address and break from loop
		    address = host;
		    break;
	    }
	}
    }

    // free structure as it is dynamically allocated
    freeifaddrs(ifaddr);
    return 0;
}

int main(int argc, char *argv[])
{
    const char* local_addr;
    if(getRPiAddress(local_addr) ==0){
        std::cout << "Local Raspberry Pi Address: " << local_addr << std::endl;
    }
}
