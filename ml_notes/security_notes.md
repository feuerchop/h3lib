## Computer and Network Security Lecture Notes

### Control hijacking

- bufferoverflow, integer overflow, use after free, mainly in C/C++ 
- Audit softwares: coverity, prefast/prefix or rewrite in type safe lang
- DEP controls in windows
- Run time checking: StackGuard
- Summary: Canaries are not full proof
	- Canaries are an important defense tool, but do not prevent all control hijacking attacks: 
		– Heap-based attacks still possible 
		– Integer overflow attacks still possible 
		– /GS by itself does not prevent Exception Handling attacks (also need SAFESEH and SEHOP)
- Or Libsafe, no need to recompile, it checks the buf length for you to prevent overflow 
- Control flow integrity systems make sure control flows as code flow graph 
	- CCFIR (2013), kBouncer (2013), FECFI (2014), CSCFI (2015) ...
- Cryptographic Control Flow Integrity (CCFI)
	- 64-bit AES-MAC and append to address, verification for this will fail attackers

### Secure architecture

- Isolation / Compartmentation design
	- assign different component with different uid with least priviledge, e.g., vagrant, mysql etc..
	- keep only one setuid and root program 
	- E.g., android application sandbox follows this design, each app runs on its own VM with its own uid. Privilideges are set at installation time 
- ACL
- Isolation
	- VMs
		- run IDS as part of VMM (protected from malware)
		- VMI: Virtual Machine Introspection, allows VMM to check Guest OS internals
		- Methods:
			- lie detector to check process of running apps in vm
			- compare hashs in memory of guest os
			- detect changes in sys_call_table
			- virus signature
	- System Call Interposition, Isolate a process in a single operating system
	- Software Fault Isolation (SFI), Isolating threads sharing same address space, e.g., browser sandbox? 
	- Jailkit, put programs in jail env. 

### Cryptography 

- Sym. and Asym. crypto
- OTP: one time pad, single key should be used only once
- MAC: message integrity
	- encrypt then mac
	- mac then encrypt
	- encrypt and mac
- Public key encryption
	- CA: trusted root certificate authority 

### Web security

Hackers tricks user to their web sites, it's different from network hacker who compromise the network communication instead. 

- image load can contain spoof, use onerror handler to check if the _src_ is actually an image 
- https used, but in DOM content there're http used still 
- Extended validation certs 
- Set-cookies: Secure=true -> then it is sent over https / httpOnly 
- Set X-Frame-Options HTTP response header, use options _DENY_, _SAMEORIGIN_, _ALLOW-FROM_ 
- Strict Transport Security (HSTS) always use https
- Always use protocol relative URLs "<img src=\"//site.com/img\">"
- TLS can also be peeked by analyzing the network traffic 
- SQL injection, see easylogin example
	- Never build SQL commands yourself !
	- Use parameterized/prepared SQL
	- Use ORM framework
- Cross-site request forgery
	- fake website to trick user to send cookie authenticator
	- Secret Validation Token 
	- referer validation
	- custom http header
- Web app firewalls, Sample products: 
	- Imperva
	- Kavado Interdo
	- F5 TrafficShield
	- Citrix NetScaler
	- CheckPoint Web Intel
- Code checking
	- whitehatsec.com
	- automatic blackbox tools: 
		- cenzic, hailstorm
		- spidynamic, webinspect
		- eEye, retina
	- web app hardening
		- webssari
		- nyguyen-tuong
- For webb app, Logout function must invalidate session token, delete from client, expire from server.

### Network security 

- IP header can be forged by machine owner, e.g., Libnet
	- so you can override the raw socket whatever you want -> DDoS attack
- Core protocols not desiged for security
	- eavesdropping, packet injection, route stealing, dns poisoning
	- more secure variants: IPSec, DNSsec, SBGP
- protect from network layer
	- 802.11i/WPA2 provides encrypted link
	- IPsec
	- mobile IPv6
- protect from application layer
	- firewall/packet filter/proxies
	- IDS such as snort, but too many false alarms
- protect from infrastructure
	- SBGP
	- DNSsec to protect authenticated DNS record data, careful about DNS rebinding attack
- DoS
	- can happen on all layers
		- 802.11b: radio jamming attack
		- 