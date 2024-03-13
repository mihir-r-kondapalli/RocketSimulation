class Body
{
	public:
		
		double MASS;
		double RADIUS;
		
		double x;
		double y;
		
		Body()
		{
			x = 0;
			y = 0;
			
			MASS = 0;
			RADIUS = 0;
		}
		
		Body(double x, double y, double MASS, double RADIUS)
		{
			this->x = x;
			this->y = y;
			
			this->MASS = MASS;
			this->RADIUS = RADIUS;
		}
};