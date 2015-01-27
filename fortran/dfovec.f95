	subroutine dfovec(n,mv,x,v_err)
		implicit none
	
		integer, intent(in) 	:: n,mv
		real(8), dimension(mv)	:: v_err
		real(8), dimension(n) 	:: x
		integer :: i

		call dfovec_iface(v_err,x,n)
	end
	
