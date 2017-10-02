/* ---------------------------------------------------------------------
 * $Id: step-20.cc 30526 2013-08-29 20:06:27Z felix.gruber $
 *
 * Copyright (C) 2005 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2005, 2006
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>

#include <typeinfo>
#include <fstream>
#include <iostream>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/tensor_function.h>

namespace project_p1
{
  using namespace dealii;

  /**
   * This class represents a function with a gradient where both of them
   * are supplied by a FunctionParser expression.
   */
  template <int dim>
    class ParsedFunctionWithGradient : public Function<dim>
    {
      public:
        ParsedFunctionWithGradient (
            unsigned int n_components_)
            :
                Function<dim>(n_components_),
                func_f(n_components_),
                func_gradient(dim * n_components_)
        {
        }

        virtual
        ~ParsedFunctionWithGradient ()
        {
        }

        void
        initialize (
            const std::string & func, const std::string & gradient)
        {
          std::map<std::string, double> constants;
          constants["pi"] = numbers::PI;
          func_f.initialize((dim==2)?"x,y":"x,y,z", func, constants);
          func_gradient.initialize((dim==2)?"x,y":"x,y,z", gradient, constants);
        }

    //* code that compiles   virtual double

        virtual void
        vector_value (
        	const Point<dim> &p, Vector<double> &value ) const
    //* code that compiles        const Point<dim> &p, const unsigned int component = 0) const

        {
        	for (unsigned int i = 0; i<this->n_components; i++)
        		value[i] = func_f.value(p,i);
        }


        virtual Tensor<1, dim>
        gradient (
            const Point<dim> &p, const unsigned int component = 0) const
        {

          Tensor<1, dim> temp;
          for (unsigned int i = 0; i < dim; ++i)
            temp[i] = func_gradient.value(p, i+component*dim);
          return temp;
        }

      private:
        //unsigned int n_components;
        FunctionParser<dim> func_f;
        FunctionParser<dim> func_gradient;
        unsigned int fe_order;

    };
  

  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    
	static void
	declare_parameters (
				ParameterHandler & prm);
	
	MixedLaplaceProblem (
				ParameterHandler & prm);  // Constructor function

	~MixedLaplaceProblem ();

    void run ();

  private:
    void refine_grid (
    		unsigned int cycle_num);
    void setup_system ();
    void assemble_system ();
    void solve ();
    void solve_direct();
    void compute_errors (int cycle_num) ;
    void output_results () const;
    void cellwise_div_error(const BlockVector<double> &calc_solution,
            Vector<double>  &output_vector,
            const Quadrature<dim> &quadrature) const;

    const unsigned int   degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        *fe;
    DoFHandler<dim>      dof_handler;
    
    ConstraintMatrix constraints;

    std::string fe_type;
    std::ofstream output_file;

    unsigned int n_cycles;
    unsigned int initial_refine;

    double grid_distort_parameter;

    bool grid_print_bool;
    bool output_convergence_rates;
    bool adaptive_refinement;

    unsigned int solver_type;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double>       solution;
    BlockVector<double>       system_rhs;
    BlockVector<double>		  solution_project;
    
    ConvergenceTable convergence_table;

    FunctionParser<dim> func_bdy_pressure;
    FunctionParser<dim> func_rhs;
    ParsedFunctionWithGradient<dim> func_reference;
  };

  template <int dim>
  void
  MixedLaplaceProblem<dim>::declare_parameters (
		  	  	  	 ParameterHandler & prm)
 	 {

	  prm.enter_subsection("equation");
	  prm.declare_entry("pressure_bdy","0",Patterns::Anything(),
			 "expression for boundary pressure.  Function of x,y (and z)");
	  prm.declare_entry("rhs","0;0;0",Patterns::Anything(),
			  "expression for the right-hand side. Function of x,y (and z)");
	  prm.declare_entry("reference","0;0;0",Patterns::Anything(),
			  "expression for reference solution and boundary values.  Function of x,y (and z)");
	  prm.declare_entry("gradient","0;0;0;0;0;0",Patterns::Anything(),
			  "expression for the gradient of the reference solution.  Function of x,y (and z)");
	  prm.leave_subsection();

	  prm.declare_entry("fe order","1", Patterns::Integer(0),
			  "order of the finite element to use.");
	  prm.declare_entry("dim","2",Patterns::Integer(1),
			  "dimension of the problem");
	  prm.declare_entry("initial_refine","2",Patterns::Integer(1),
			  "initial global refinement");
	  prm.declare_entry("n_cycles","3",Patterns::Integer(1),
			  "program loop cycles");
	  prm.declare_entry("grid_distort_parameter","0",Patterns::Double(0),
			  "grid distortion parameter");
	  prm.declare_entry("grid_print_bool","false",Patterns::Bool(),
			  "grid print out option");
	  prm.declare_entry("output_convergence_rates","false",Patterns::Bool(),
			  "output convergence rates to text file");
	  prm.declare_entry("solver_type","1",Patterns::Integer(0),
			  "set solver type (0 for direct solver 1 schur complement)");
	  prm.declare_entry("adaptive_refinement","false",Patterns::Bool(),
			  "adaptive refinement option");
	  prm.declare_entry("fe_type","RT",Patterns::Anything(),
			  "set the finite element type");

  	 }

  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };


  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1.;
      }
  }

  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem (
  	  ParameterHandler & prm)
    :
    degree (prm.get_integer("fe order")),
    dof_handler (triangulation),
    func_bdy_pressure(1),
    func_rhs(dim+1),
    func_reference(dim+1)
  {
	  if (prm.get("fe_type")=="RT")

		  fe = new FESystem<dim>(FE_RaviartThomas<dim>(degree), 1,
				      		FE_DGQ<dim>(degree),1);

	  else if (prm.get("fe_type")=="TH")

		  fe = new FESystem<dim>(FE_Q<dim>(degree),dim,
		    	FE_Q<dim>(degree-1),1);

	  else
	 	  fe = new FESystem<dim>(FE_BDM<dim>(degree), 1,
				     FE_DGP<dim>(degree-1), 1);

	  // the following lines tests the proper fe is being used..
	  //	  std::cout << "FINITE ELEMENT TYPE" << fe->get_name() << std::endl;

	  fe_type = prm.get("fe_type");
	  n_cycles = prm.get_integer("n_cycles");
	  initial_refine = prm.get_integer("initial_refine");
	  grid_distort_parameter = prm.get_double("grid_distort_parameter");
	  grid_print_bool = prm.get_bool("grid_print_bool");
	  output_convergence_rates = prm.get_bool("output_convergence_rates");
	  adaptive_refinement = prm.get_bool("adaptive_refinement");
	  solver_type = prm.get_integer("solver_type");

	  prm.enter_subsection("equation");
	  std::map<std::string,double> constants;
	  constants["pi"] = numbers::PI;
	  func_bdy_pressure.initialize((dim==2)?"x,y":"x,y,z",prm.get("pressure_bdy"),constants);
	  func_rhs.initialize((dim==2)?"x,y":"x,y,z",prm.get("rhs"),constants);
	  func_reference.initialize(prm.get("reference"),prm.get("gradient"));
	  prm.leave_subsection();


  }

template <int dim>
MixedLaplaceProblem<dim>::~MixedLaplaceProblem ()
{
	dof_handler.clear();
}


 template <int dim>
 void MixedLaplaceProblem<dim>::refine_grid(unsigned int cycle_num)
	{

	 if (adaptive_refinement == true)
	 {
	 Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

	 FEValuesExtractors::Scalar pressure(dim);
	 //FEValuesExtractors::Vector<dim> velocity(dim-1);

	 KellyErrorEstimator<dim>::estimate (dof_handler,
			 	 	 	 	 	 	 	 QGauss<dim-1>(dim+1),
			 	 	 	 	 	 	 	 typename FunctionMap<dim>::type(),
			 	 	 	 	 	 	 	 solution,
			 	 	 	 	 	 	 	 estimated_error_per_cell,
			 	 	 	 	 	 	 	 fe->component_mask(pressure)
			 	 	 	 	 	 	      );

	 GridRefinement::refine_and_coarsen_fixed_number (triangulation,
			 	 	 	 	 	 	 	 	 	 	   estimated_error_per_cell,
			 	 	 	 	 	 	 	 	 	 	   0.3,0.03);

	 triangulation.execute_coarsening_and_refinement();

	 }

	 else{

		 triangulation.refine_global (1);
	 }

	 //GridTools::distort_random(grid_distort_parameter, triangulation, false);


	if (grid_print_bool != 0){

	GridOut grid_out;
	std::ostringstream filename;


	filename << "grid-"
			 << cycle_num
			 << ".eps";
	std::ofstream output (filename.str().c_str());

	grid_out.write_eps (triangulation,output);
/*
	std::cout << " written to " << "grid-1.eps"
							<< std::endl
							<< std::endl;
							*/
	}


	}


  template <int dim>
  void MixedLaplaceProblem<dim>::setup_system ()
  {
    dof_handler.distribute_dofs (*fe);
    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

    unsigned int n_u;
    unsigned int n_p;

    if (fe_type == "TH"){
    	n_u = dofs_per_component[0]+dofs_per_component[1];
    	n_p = dofs_per_component[dim];
    }
    else{
    	n_u = dofs_per_component[0];
        n_p = dofs_per_component[dim];
    }


    // put a conditional statement in here for adaptive refinement?

    constraints.clear();

    DoFTools::make_hanging_node_constraints (dof_handler,
    											constraints);
    constraints.close();
    /*
    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')'
              << std::endl;
	*/

    // const unsigned int
    // n_couplings = dof_handler.max_couplings_between_dofs();

    system_matrix.clear();

    BlockCompressedSimpleSparsityPattern c_sparsity(2,2);

    c_sparsity.block(0,0).reinit (n_u, n_u);
    c_sparsity.block(1,0).reinit (n_p, n_u);
    c_sparsity.block(0,1).reinit (n_u, n_p);
    c_sparsity.block(1,1).reinit (n_p, n_p);

    c_sparsity.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler,
    								c_sparsity,
    								constraints,
    								/* keep_constrained_dofs =*/ false);

    sparsity_pattern.copy_from(c_sparsity);

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();

    solution_project.reinit (2);
    solution_project.block(0).reinit (n_u);
    solution_project.block(1).reinit (n_p);
    solution_project.collect_sizes ();

  }



  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system ()
  {
    QGauss<dim>   quadrature_formula(degree+2);
    QGauss<dim-1> face_quadrature_formula(degree+2);

    FEValues<dim> fe_values (*fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe->dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);


    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    //const RightHandSide<dim>          right_hand_side;
    //const PressureBoundaryValues<dim> pressure_boundary_values;
    const KInverse<dim>               k_inverse;

    std::vector<Vector<double> > rhs_values (n_q_points,Vector<double>(dim+1));
    std::vector<double> boundary_values (n_face_q_points);
    std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;

        func_rhs.vector_value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
        k_inverse.value_list (fe_values.get_quadrature_points(),
                              k_inverse_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);
              const double        div_phi_i_u = fe_values[velocities].divergence (i, q);
              const double        phi_i_p     = fe_values[pressure].value (i, q);

              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
                  const double        div_phi_j_u = fe_values[velocities].divergence (j, q);
                  const double        phi_j_p     = fe_values[pressure].value (j, q);

                  local_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                        - div_phi_i_u * phi_j_p
                                        - phi_i_p * div_phi_j_u)
                                       * fe_values.JxW(q);
                }
              for (int c=0;c < dim; c++){
              local_rhs(i) += fe_values.shape_value_component(i,q,c) *
                              rhs_values[q](c) *
                              fe_values.JxW(q);
              }

            }

        for (unsigned int face_no=0;
             face_no<GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->at_boundary(face_no))
            {
              fe_face_values.reinit (cell, face_no);

              func_bdy_pressure
              .value_list (fe_face_values.get_quadrature_points(),
                           boundary_values);

              for (unsigned int q=0; q<n_face_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i){
                  local_rhs(i) += -(fe_face_values[velocities].value (i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values[q] *
                                    fe_face_values.JxW(q));

                }

            }

        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
           				local_rhs,
        				local_dof_indices,
        				system_matrix,system_rhs);
      }


  }


  template <int dim>
  void MixedLaplaceProblem<dim>::solve_direct ()
  {
	  SparseDirectUMFPACK A_direct;
	  A_direct.initialize(system_matrix);
	  A_direct.vmult (solution,system_rhs);

	  QGauss<dim>   quad_formula(degree+2);
	  VectorTools::project(dof_handler,constraints,quad_formula,func_reference,solution_project);

	    constraints.distribute(solution);
  }


  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement (const BlockSparseMatrix<double> &A,
                     const IterativeInverse<Vector<double> > &Minv);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
    const SmartPointer<const IterativeInverse<Vector<double> > > m_inverse;

    mutable Vector<double> tmp1, tmp2;
  };


  SchurComplement::SchurComplement (const BlockSparseMatrix<double> &A,
                                    const IterativeInverse<Vector<double> > &Minv)
    :
    system_matrix (&A),
    m_inverse (&Minv),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void SchurComplement::vmult (Vector<double>       &dst,
                               const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    m_inverse->vmult (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }



  class ApproximateSchurComplement : public Subscriptor
  {
  public:
    ApproximateSchurComplement (const BlockSparseMatrix<double> &A);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;
    void Tvmult (Vector<double>       &dst,
                 const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;

    mutable Vector<double> tmp1, tmp2;
  };


  ApproximateSchurComplement::ApproximateSchurComplement (const BlockSparseMatrix<double> &A)
    :
    system_matrix (&A),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void ApproximateSchurComplement::vmult (Vector<double>       &dst,
                                          const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }


  void ApproximateSchurComplement::Tvmult (Vector<double>       &dst,
                                           const Vector<double> &src) const
  {
    vmult (dst, src);
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::solve ()
  {
    PreconditionIdentity identity;
    IterativeInverse<Vector<double> > m_inverse;
    m_inverse.initialize(system_matrix.block(0,0), identity);
    m_inverse.solver.select("cg");
    static ReductionControl inner_control(1000, 0., 1.e-13);
    m_inverse.solver.set_control(inner_control);

    QGauss<dim>   quad_formula(degree+2);
    VectorTools::project(dof_handler,constraints,quad_formula,func_reference,solution_project);

    Vector<double> tmp (solution.block(0).size());

    {
      Vector<double> schur_rhs (solution.block(1).size());

      m_inverse.vmult (tmp, system_rhs.block(0));
      system_matrix.block(1,0).vmult (schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);


      SchurComplement
      schur_complement (system_matrix, m_inverse);

      ApproximateSchurComplement
      approximate_schur_complement (system_matrix);

      IterativeInverse<Vector<double> >
      preconditioner;
      preconditioner.initialize(approximate_schur_complement, identity);
      preconditioner.solver.select("cg");
      preconditioner.solver.set_control(inner_control);


      SolverControl solver_control (solution.block(1).size(),
                                    1e-12*schur_rhs.l2_norm());
      SolverCG<>    cg (solver_control);

      cg.solve (schur_complement, solution.block(1), schur_rhs,
                preconditioner);

    }

    {
      system_matrix.block(0,1).vmult (tmp, solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      m_inverse.vmult (solution.block(0), tmp);
    }

    constraints.distribute(solution);
  }

  template <int dim>
  void
  MixedLaplaceProblem<dim>::cellwise_div_error(const BlockVector<double> &calc_solution,
                                         Vector<double>  &output_vector,
                                         const Quadrature<dim> &quadrature) const
  {
    output_vector = 0;
    static MappingQ<dim> mapping(1);

    FEValues<dim> fe_values(mapping,
                            dof_handler.get_fe(),
                            quadrature,
                            update_gradients | update_JxW_values);
    const FEValuesExtractors::Vector velocities (0);


    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    const unsigned int n_q_points = quadrature.size();

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    unsigned int cellindex=0;
    for (; cell!=endc; ++cell, ++cellindex)
      {
        fe_values.reinit(cell);
        cell->get_dof_indices (local_dof_indices);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double tmp = 0;
            for (unsigned int i=0; i < dofs_per_cell; ++i)
              {
                tmp += fe_values[velocities].divergence(i, q) *
                    calc_solution(local_dof_indices[i]);
              }
            output_vector(cellindex) += tmp * tmp * fe_values.JxW(q);
//            std::cout << tmp << " " << fe_values.JxW(q) <<std::endl;
          }
//        std::cout << output_vector(cellindex) <<std::endl;
        output_vector(cellindex) = std::sqrt(output_vector(cellindex));
      }
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors (int cycle_num)
  {
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0, dim), dim+1);

    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QGauss<dim>  quadrature(degree+2);

    //QTrapez<1>     q_trapez;
    //QIterated<dim> quadrature (q_trapez, degree+2);

    VectorTools::integrate_difference (dof_handler, solution, func_reference,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, solution_project, func_reference,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_project_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, solution, func_reference,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler,solution_project,func_reference,
    									cellwise_errors,quadrature,
    									VectorTools::L2_norm,
    									&velocity_mask);
    const double u_project_l2_error = cellwise_errors.l2_norm();

    MixedLaplaceProblem::cellwise_div_error(solution,cellwise_errors, quadrature);
    const double u_l2div_error = cellwise_errors.l2_norm();

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
//    		  << ",   ||e_p_project||_L2" << p_project_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error
//              << ",   ||e_u_project||_L2 = " << u_project_l2_error
              //  ***** It is strange that the divergence error returns NaN
              //  ****** Something to investigate further
              << ",   ||e_div||_L2 = " << u_l2div_error
              << std::endl;

    convergence_table.add_value("cycle", cycle_num);
    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("u_L2",u_l2_error);
    convergence_table.add_value("p_L2",p_l2_error);

    if (output_convergence_rates != 0){
  	std::ostringstream fn;
  	fn << "dim" <<dim << fe_type << degree << "ConvergeData" << ".txt";

    output_file.open(fn.str().c_str(),std::ios_base::app);
    		output_file << dof_handler.n_dofs() << ";" << u_l2_error << ";" << p_l2_error << ";"
    				<< u_l2div_error << "\n";
    output_file.close();
    }

  }

  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
    //std::vector<std::string> solution_names;
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("p");
//    switch (dim)
//      {
//      case 2:
//        solution_names.push_back ("u");
//        solution_names.push_back ("v");
//        solution_names.push_back ("p");
//        break;
//
//      case 3:
//        solution_names.push_back ("u");
//        solution_names.push_back ("v");
//        solution_names.push_back ("w");
//        solution_names.push_back ("p");
//        break;
//
//      default:
//        Assert (false, ExcNotImplemented());
//      }
//

    DataOut<dim> data_out;
    DataOut<dim> data_out_2;

    std::vector<DataComponentInterpretation::DataComponentInterpretation> dci
    (dim+1,DataComponentInterpretation::component_is_part_of_vector);
    dci[dim] = DataComponentInterpretation::component_is_scalar;

//    data_out.attach_dof_handler (dof_handler);

    data_out_2.add_data_vector (dof_handler,
    		solution_project,
    		solution_names,
    		dci);

    data_out.add_data_vector (dof_handler,
    		solution,
    		solution_names,
    		dci);

    data_out.build_patches (degree+2);
    data_out_2.build_patches (degree+2);

    std::ofstream output_1 ("solution_project.vtk");
    data_out_2.write_vtk (output_1);

    std::ofstream output ("solution.vtk");
    data_out.write_vtk (output);
  }



  template <int dim>
  void MixedLaplaceProblem<dim>::run ()
  {

	  if (output_convergence_rates != 0){
	  	std::ostringstream fn;
	  	fn << "dim" <<dim << fe_type << degree << "ConvergeData" << ".txt";

	    output_file.open(fn.str().c_str());
	    		output_file << "# The following gives convergence data for dimension "
	    					<< dim
	    					<< " using finite element "
	    					<< fe_type
	    					<< " "
	    					<< degree
	    					<< "\n"
	    					<< "col 1 -- dof, col 2 -- u_l2_error, col 3 -- p_l2_error, col 4 -- div_error \n";
	    output_file.close();
	  }

	for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
	{
		if(cycle == 0) {
			//Point<dim> p (0,0);
			//GridGenerator::hyper_ball(triangulation,p,1);
			GridGenerator::hyper_cube(triangulation,-1,1);
			triangulation.refine_global(initial_refine);
			GridTools::distort_random(grid_distort_parameter, triangulation, false);

			if (grid_print_bool != 0){

			std::ofstream out ("grid-1.eps");
			GridOut grid_out;
			grid_out.write_eps(triangulation,out);
	/*
			std::cout << " written to " << "grid-1.eps"
						<< std::endl
						<< std::endl;
						*/
			}
					}
		else
		refine_grid(cycle+1);


    setup_system();
    assemble_system ();
    if (solver_type==0){
    	solve ();
    }
    if (solver_type==1){
    	solve_direct ();
    }
    compute_errors (initial_refine + cycle);
    output_results ();

	}

    convergence_table.set_precision("u_L2",3);
    convergence_table.set_precision("p_L2",3);

    convergence_table.set_scientific("u_L2",true);
    convergence_table.set_scientific("p_L2",true);

    convergence_table.set_tex_caption("cells","\\# cells");
    convergence_table.set_tex_caption("dofs","\\# dof");
    convergence_table.set_tex_caption("u_L2","u_L2 error");
    convergence_table.set_tex_caption("p_L2","p_L2 error");

    convergence_table.set_tex_format("cells","r");
    convergence_table.set_tex_format("dofs","r");

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    /*
    std::string error_filename = "error";
    error_filename + ".tex";
    std::ofstream error_table_file(error_filename.c_str());
    convergence_table.write_tex(error_table_file);
	*/

    if (adaptive_refinement == 0){
    	//convergence_table.add_column_to_supercolumn("cycles","n cells");
    	//convergence_table.add_column_to_supercolumn("cells","n cells");

    	std::vector<std::string> new_order;
    	new_order.push_back ("cells");
    	new_order.push_back ("u_L2");
    	new_order.push_back ("p_L2");
    	convergence_table.set_column_order (new_order);

    	convergence_table.evaluate_convergence_rates("u_L2", ConvergenceTable::reduction_rate_log2);
    	convergence_table.evaluate_convergence_rates("p_L2", ConvergenceTable::reduction_rate_log2);

    	std::cout << std::endl;

    	convergence_table.write_text(std::cout);


    	std::string error_filename = "error";
        error_filename + ".tex";
        std::ofstream error_table_file(error_filename.c_str());
        convergence_table.write_tex(error_table_file);
    }

  }
}



int main (
		int argc, char *argv[])
{
  if (argc != 2)
  {
	  std::cerr <<"  usuage: ./step-7 <parameter-file.prm>" << std::endl;
	  return -1;
	  
  }
  
  try
    {
      using namespace dealii;
      using namespace project_p1;

      deallog.depth_console (0);

      ParameterHandler prm;
      MixedLaplaceProblem<3>::declare_parameters(prm);
      
      prm.read_input(argv[1]);
      
      int dim = prm.get_integer("dim");

      if (dim==2) {
    	  MixedLaplaceProblem<2> mixed_laplace_problem(prm);
    	  mixed_laplace_problem.run ();
      }
      else if (dim==3) {
    	  MixedLaplaceProblem<3> mixed_laplace_problem(prm);
    	  mixed_laplace_problem.run ();
      }
      }

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
